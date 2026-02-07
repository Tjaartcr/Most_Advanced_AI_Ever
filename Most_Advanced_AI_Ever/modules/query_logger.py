
import os
import json
from datetime import datetime
from typing import List, Dict, Any

LOG_DIR = "user_logs"
MAX_LOG_ENTRIES = 1000  # Set to None for unlimited logs

os.makedirs(LOG_DIR, exist_ok=True)


def _get_log_path(user: str) -> str:
    return os.path.join(LOG_DIR, f"{user}.json")


def _make_backup_filename(path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{path}.{timestamp}.corrupt"


def backup_corrupt_file(path: str):
    """
    Rename the current corrupt file to a timestamped .corrupt backup.
    """
    try:
        backup_path = _make_backup_filename(path)
        os.rename(path, backup_path)
        print(f"[Logger] ⚠️ Corrupt file backed up as {backup_path}")
    except Exception as e:
        print(f"[Logger Error] Failed to backup corrupt file: {e}")


def try_partial_recovery(path: str) -> List[Dict[str, Any]]:
    """
    Read the file in a lenient way, extract the first [...] JSON array
    and return its contents. If that fails, returns [].
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []

        snippet = raw[start : end + 1]
        data = json.loads(snippet)
        print(f"[Logger] ✅ Partial recovery successful: {len(data)} entries")
        return data
    except Exception as e:
        print(f"[Logger Error] Partial recovery failed: {e}")
        return []


def _load_logs(path: str) -> List[Dict[str, Any]]:
    """
    Load logs from path in a defensive manner:
     - if file missing or empty -> []
     - try json.load
     - on JSONDecodeError -> attempt partial recovery and write recovered JSON back (after backing up corrupt file)
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as jde:
        # Try to salvage a partial array
        print(f"[Logger] JSON decode error for {path}: {jde}. Attempting partial recovery...")
        recovered = try_partial_recovery(path)
        try:
            # Backup original corrupt file first
            backup_corrupt_file(path)
            if recovered:
                # write recovered data back to the original path as valid JSON
                _write_logs(path, recovered)
                print(f"[Logger] Recovered JSON written to {path}")
                return recovered
            else:
                print(f"[Logger] No recoverable JSON found in {path}; returning empty list")
                return []
        except Exception as e:
            print(f"[Logger] Failed to write recovered logs for {path}: {e}")
            return []
    except Exception as e:
        print(f"[Logger] Unexpected error loading logs from {path}: {e}")
        return []


def _write_logs(path: str, logs: List[Dict[str, Any]]):
    """
    Write logs as JSON using json.dump. This preserves newlines inside strings (they are escaped as \n
    in the file, and json.load will decode them back into real newline characters).
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Logger Error] Failed to write logs to {path}: {e}")
        raise


def _normalize_entry(user: str, entry: Any, models_override: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Ensure an entry dict has the canonical fields we expect:
      - username
      - timestamp (iso)
      - query (string)
      - response (string or None)
      - models (dict)

    Accepts either a dict 'entry' or a string (treated as query text).
    """
    if isinstance(entry, dict):
        e = dict(entry)  # shallow copy
    else:
        # treat a non-dict as the raw query string
        e = {"query": str(entry)}

    # username
    e.setdefault("username", user)

    # timestamp
    if not e.get("timestamp"):
        e["timestamp"] = datetime.now().isoformat()

    # query: if nested shapes exist (older code), try to flatten
    # support legacy where entry may have 'query' as dict with 'query' and 'response'
    if isinstance(e.get("query"), dict):
        qdict = e["query"]
        e["query"] = qdict.get("query", "")
        # prefer top-level response if present, otherwise use nested
        if not e.get("response") and isinstance(qdict.get("response"), str):
            e["response"] = qdict.get("response")

    # ensure query and response keys exist
    e.setdefault("query", "")
    e.setdefault("response", None)

    # models: merge priority: models_override > entry['models'] > {}
    entry_models = e.get("models") if isinstance(e.get("models"), dict) else {}
    merged = dict(entry_models)
    if isinstance(models_override, dict):
        merged.update(models_override)
    # attach back
    e["models"] = merged

    return e


def log_user_query(user: str, query: Any, models: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Append a new query entry; accepts either:
      - query: string (text)
      - query: dict (full entry)
    'models' optionally merges into the stored entry.
    On JSON errors, back up + attempt partial recovery before writing.

    Returns the saved entry (so frontend can immediately display model + query).
    """
    path = _get_log_path(user)

    # Normalize the incoming data into canonical entry shape
    entry = _normalize_entry(user, query, models_override=models)

    # Ensure query is always a string (defensive)
    if isinstance(entry.get("query"), dict):
        # flatten if something slipped through
        entry["query"] = entry["query"].get("query", "")
    entry["query"] = str(entry.get("query", ""))

    try:
        logs = _load_logs(path)
        if not isinstance(logs, list):
            logs = []

        logs.append(entry)

        # Trim to configured max
        if MAX_LOG_ENTRIES:
            logs = logs[-MAX_LOG_ENTRIES:]

        # Persist using your writer helper (keeps behaviour consistent)
        _write_logs(path, logs)

        print(f"[Logger] ✅ Query logged for {user} (ts={entry['timestamp']})")

        # return the saved entry so frontend can show it immediately
        return entry

    except Exception as ex:
        # Keep behaviour predictable for callers: return constructed entry even on error
        print(f"[Logger Error] log_user_query: {ex}")
        # attempt fallback recovery path
        try:
            # try partial recovery and append
            recovered = try_partial_recovery(path)
            backup_corrupt_file(path)
            if not isinstance(recovered, list):
                recovered = []
            recovered.append(entry)
            if MAX_LOG_ENTRIES:
                recovered = recovered[-MAX_LOG_ENTRIES:]
            _write_logs(path, recovered)
            return entry
        except Exception as e2:
            print(f"[Logger Error] fallback write failed: {e2}")
            return entry


def log_user_response(user: str, response: str, models: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Attach a response (multiline string allowed) to the last saved query for 'user', or append a new entry.
    This function *preserves* the response string as-is (no json.dumps on the string), letting json.dump
    handle encoding when writing to file (so newlines remain).
    """
    path = _get_log_path(user)
    entry = None

    try:
        logs = _load_logs(path)
        if not isinstance(logs, list):
            logs = []

        if logs:
            last = logs[-1]
            # If last entry appears to be a query without response, set response there
            if not last.get("response"):
                last["response"] = response
                # merge models if provided
                if isinstance(models, dict):
                    last_models = last.get("models") or {}
                    last_models.update(models)
                    last["models"] = last_models
                entry = last
            else:
                # otherwise append a new entry containing this response
                new_entry = _normalize_entry(user, {"query": "", "response": response}, models_override=models)
                logs.append(new_entry)
                entry = new_entry
        else:
            # no logs exist; create a new one
            new_entry = _normalize_entry(user, {"query": "", "response": response}, models_override=models)
            logs.append(new_entry)
            entry = new_entry

        # Trim and persist
        if MAX_LOG_ENTRIES:
            logs = logs[-MAX_LOG_ENTRIES:]
        _write_logs(path, logs)

        print(f"[Logger] ✅ Response logged for {user} (ts={entry['timestamp']})")
        return entry

    except Exception as ex:
        print(f"[Logger Error] log_user_response: {ex}")
        # try fallback recovery + write
        try:
            recovered = try_partial_recovery(path)
            backup_corrupt_file(path)
            if not isinstance(recovered, list):
                recovered = []
            # attach entry to recovered
            maybe_entry = _normalize_entry(user, {"query": "", "response": response}, models_override=models)
            recovered.append(maybe_entry)
            if MAX_LOG_ENTRIES:
                recovered = recovered[-MAX_LOG_ENTRIES:]
            _write_logs(path, recovered)
            return maybe_entry
        except Exception as e2:
            print(f"[Logger Error] fallback write in log_user_response failed: {e2}")
            return {"username": user, "timestamp": datetime.now().isoformat(), "response": response, "models": models or {}}


def log_user_response(user, message, models=None):
    """
    Attach a response to the last query entry; on JSON errors,
    back up + attempt partial recovery before writing.

    Keeps the legacy nested `last['query']['response']` AND writes
    a top-level `last['response']` so both old and new frontends work.
    Optionally merges a `models` dict into the entry.
    """
    path = _get_log_path(user)

    try:
        # Load existing
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []

        if not logs:
            print(f"[Logger Warning] No existing entry to attach response for {user}.")
            return

        last = logs[-1]

        # If last["query"] is a string (legacy), preserve it and copy into the nested shape
        if not isinstance(last.get("query"), dict):
            prev_q = last.get("query") if last.get("query") is not None else ""
            # keep a safe copy for older frontends that expect a top-level string
            last["query_text"] = prev_q
            last["query"] = {
                "timestamp": last.get("timestamp", datetime.now().isoformat()),
                "query": prev_q,          # <- preserve original query text here
                "response": message,
                "models": {}
            }
        else:
            # nested dict exists; update response and timestamp
            last["query"]["response"] = message
            last["query"]["timestamp"] = datetime.now().isoformat()

        # Also set top-level response for newer frontends
        last["response"] = message

        # Merge provided models into both nested and top-level models
        if isinstance(models, dict):
            # nested
            last["query"].setdefault("models", {})
            last["query"]["models"].update(models)
            # top-level
            last.setdefault("models", {})
            last["models"].update(models)

        # Write back
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        print(f"[Logger] ✅ Response logged for {user}")

    except json.JSONDecodeError as e:
        print(f"[Logger Error] JSON decode error in response for {user}: {e}")
        backup_corrupt_file(path)
        logs = try_partial_recovery(path)
        if not logs:
            print(f"[Logger Warning] No entries recovered for response logging of {user}.")
            return
        last = logs[-1]

        if not isinstance(last.get("query"), dict):
            prev_q = last.get("query") if last.get("query") is not None else ""
            last["query_text"] = prev_q
            last["query"] = {
                "timestamp": last.get("timestamp", datetime.now().isoformat()),
                "query": prev_q,
                "response": message,
                "models": {}
            }
        else:
            last["query"]["response"] = message
            last["query"]["timestamp"] = datetime.now().isoformat()

        last["response"] = message

        if isinstance(models, dict):
            last["query"].setdefault("models", {})
            last["query"]["models"].update(models)
            last.setdefault("models", {})
            last["models"].update(models)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"[Logger Error] log_user_response: {e}")


def read_logs(user):
    """
    Return the full log list; on JSON errors, back up +
    attempt partial recovery and overwrite the file with the recovered subset.
    """
    path = _get_log_path(user)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        print(f"[Logger Error] Failed to read JSON for {user}: {e}")
        backup_corrupt_file(path)
        recovered = try_partial_recovery(path)
        _write_logs(path, recovered)
        return recovered

    except Exception as e:
        print(f"[Logger Error] read_logs: {e}")
        return []

