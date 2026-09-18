"""Best-effort Codex context warning hook.

The rollout transcript is convenient but explicitly not a stable hook API.
This script therefore fails open and reports UNKNOWN whenever evidence is
missing, stale, malformed, or belongs to another session.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

MAX_AGE_SECONDS = 30 * 60


def _unknown(reason: str) -> dict[str, Any]:
    return {"status": "unknown", "reason": reason}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _is_compaction(row: dict[str, Any]) -> bool:
    if row.get("type") == "compacted":
        return True
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return False
    kind = payload.get("type")
    return kind in {"compacted", "context_compaction", "contextCompaction"}


def inspect(transcript_path: str | Path | None, session_id: str | None,
            *, now: datetime | None = None, input_threshold: int = 80000) -> dict[str, Any]:
    if not transcript_path or not session_id:
        return _unknown("missing transcript path or session id")
    path = Path(transcript_path)
    if not path.is_file():
        return _unknown("transcript unavailable")

    try:
        raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [json.loads(line) for line in raw_lines]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _unknown("transcript unreadable or incomplete")

    observed_session = None
    latest_usage = None
    compaction_key = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        payload = row.get("payload")
        if row.get("type") == "session_meta" and isinstance(payload, dict):
            observed_session = payload.get("id")
        if _is_compaction(row):
            latest_usage = None
            compaction_key = row.get('timestamp') or str(index)
            continue
        if (row.get("type") == "event_msg" and isinstance(payload, dict)
                and payload.get("type") == "token_count"):
            latest_usage = row

    if observed_session != session_id:
        return _unknown("transcript does not match this session")
    if latest_usage is None:
        return _unknown("no post-compaction token measurement")

    payload = latest_usage.get("payload", {})
    info = payload.get("info")
    if not isinstance(info, dict):
        return _unknown("token measurement has no info")
    last = info.get("last_token_usage")
    if not isinstance(last, dict):
        return _unknown("token measurement has no last request")
    used = last.get("total_tokens")
    last_input = last.get("input_tokens")
    window = info.get("model_context_window")
    if (not _number(used) or not _number(window) or used < 0 or window <= 0
            or used > window):
        return _unknown("invalid token measurement")

    measured_at = _parse_time(latest_usage.get("timestamp"))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if measured_at is None:
        return _unknown("token measurement has no timestamp")
    age = (current - measured_at).total_seconds()
    if age > MAX_AGE_SECONDS or age < -300:
        return _unknown("token measurement is stale")

    remaining = max(0.0, min(100.0, 100.0 * (window - used) / window))
    if remaining < 10:
        status = "handoff"
    elif remaining <= 20:
        status = "checkpoint"
    elif remaining <= 30:
        status = "caution"
    else:
        status = "normal"
    return {
        "status": status,
        "remaining_percent_estimate": round(remaining),
        "last_request_tokens": used,
        "model_context_window": window,
        "measured_at": measured_at.isoformat(),
        "last_input_tokens": last_input if _number(last_input) and 0 <= last_input <= used else None,
        "expensive_input": (last_input >= input_threshold if _number(last_input)
                            and 0 <= last_input <= used else None),
        "compaction_key": compaction_key,
    }


def should_notify(result: dict[str, Any], session_id: str, state_dir: Path,
                  stage: str = '') -> bool:
    """One advisory per threshold/stage; repeat on >=25% input growth or higher severity."""
    key = hashlib.sha256(session_id.encode('utf-8')).hexdigest()
    state_path = Path(state_dir) / (key + '.json')
    current = {"status": result['status'], "input": result.get('last_input_tokens') or 0,
               "expensive": bool(result.get('expensive_input')), "stage": stage,
               "compaction": result.get('compaction_key')}
    advisory = current['status'] != 'normal' or current['expensive']
    try:
        try:
            previous = json.loads(state_path.read_text(encoding='utf-8'))
            if not isinstance(previous, dict):
                previous = {}
        except (OSError, ValueError):
            previous = {}
        if not stage:
            current['stage'] = previous.get('stage', '')
        ranks = {'normal': 0, 'unknown': 0, 'caution': 1, 'checkpoint': 2, 'handoff': 3}
        changed = (not previous or current['stage'] != previous.get('stage')
                   or current['compaction'] != previous.get('compaction')
                   or (current['status'] == 'unknown') != (previous.get('status') == 'unknown')
                   or ranks[current['status']] > ranks.get(previous.get('status'), 0)
                   or current['expensive'] and not previous.get('expensive')
                   or current['input'] >= max(1, previous.get('input', 0)) * 1.25)
        notify = advisory and changed
        if notify or not advisory:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            # Replace atomically, without putting session IDs or prompts into filenames/content.
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=state_path.parent,
                                             suffix='.tmp', delete=False) as stream:
                json.dump(current, stream)
                temporary = stream.name
            try:
                os.replace(temporary, state_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return notify
    except (OSError, ValueError, TypeError):
        # A broken advisory-state store must never block the user's work.
        return advisory


def hook(event: dict[str, Any], *, input_threshold: int = 80000,
         state_dir: Path | None = None, stage: str = '') -> dict[str, Any]:
    if event.get("hook_event_name") != "UserPromptSubmit":
        return {}
    result = inspect(event.get("transcript_path"), event.get("session_id"), input_threshold=input_threshold)
    status = result["status"]
    if event.get('session_id'):
        directory = state_dir or Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'state/context-watch'
        if not should_notify(result, event['session_id'], directory, stage):
            return {}
    if status == "unknown":
        message = "CONTEXT ESTIMATE UNKNOWN. Do not infer that ample context remains; use the host's /status or status line."
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": message}}
    if status == "normal" and not result.get('expensive_input'):
        return {}

    remaining = result["remaining_percent_estimate"]
    actions = {
        "normal": "At the next stage boundary, consider compacting or a short $end handoff.",
        "caution": "Avoid starting an unusually large new phase without a checkpoint.",
        "checkpoint": "Create a checkpoint now; choose /compact for the same task or $end for a clean-session handoff.",
        "handoff": "Finish the current safe atomic action, then use $end for handoff.",
    }
    message = f"Approximate context remaining: {remaining}%. {actions[status]}"
    if result.get('expensive_input'):
        message += f" Last input: {result['last_input_tokens']} tokens (advisory threshold {input_threshold}; not a billing limit)."
    return {
        "systemMessage": message,
        "hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": message},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Local context advisory; no model/API calls.')
    parser.add_argument('--transcript')
    parser.add_argument('--session-id')
    parser.add_argument('--stage', default='', help='Optional stage label; a new label resets the advisory')
    parser.add_argument('--input-threshold', type=int, default=80000)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--json', action='store_true', help='Read-only inspection, without notification state')
    args = parser.parse_args()
    if args.input_threshold <= 0:
        parser.error('--input-threshold must be positive')
    if bool(args.transcript) != bool(args.session_id):
        parser.error('--transcript and --session-id must be supplied together')
    try:
        event = ({'hook_event_name': 'UserPromptSubmit', 'transcript_path': args.transcript,
                  'session_id': args.session_id} if args.transcript else json.load(sys.stdin))
        if not isinstance(event, dict):
            event = {}
        result = (inspect(event.get('transcript_path'), event.get('session_id'), input_threshold=args.input_threshold)
                  if args.json else hook(event, input_threshold=args.input_threshold,
                                         state_dir=args.state_dir, stage=args.stage))
        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
