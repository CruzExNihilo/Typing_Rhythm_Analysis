from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


VALID_EVENT_TYPES = {"keydown", "keyup", "input", "timer"}


def build_session_name(sequence: int, date_str: Optional[str] = None) -> str:
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    return f"{sequence}_{date_str}.json"


def normalize_session_file(source: Path, session_dir: Path, sequence: Optional[int] = None, date_str: Optional[str] = None) -> Path:
    source = source.resolve()
    session_dir = session_dir.resolve()

    if sequence is None:
        sequence = _infer_sequence_from_existing_files(session_dir)
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    target_name = build_session_name(sequence, date_str)
    target = session_dir / target_name

    if target.exists() and target.resolve() != source.resolve():
        raise FileExistsError(f"Target file already exists: {target}")

    if source != target:
        source.replace(target)

    return target


def _infer_sequence_from_existing_files(session_dir: Path) -> int:
    existing = [p for p in session_dir.glob("*.json") if p.is_file()]
    if not existing:
        return 1

    numbers = []
    for path in existing:
        stem = path.stem
        if "_" in stem:
            prefix = stem.split("_", 1)[0]
        else:
            prefix = stem
        try:
            numbers.append(int(prefix))
        except ValueError:
            continue
    if not numbers:
        return 1
    return max(numbers) + 1


def _is_valid_session_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False

    events = payload.get("events")
    if not isinstance(events, list):
        return False

    for event in events:
        if not isinstance(event, dict):
            return False
        if "type" not in event or event["type"] not in VALID_EVENT_TYPES:
            return False
        if "data" not in event or not isinstance(event["data"], dict):
            return False

    return True


def organize_sessions(session_dir: Path, date_str: Optional[str] = None) -> list[Path]:
    session_dir = session_dir.resolve()
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    files = sorted(session_dir.glob("*.json"), key=lambda p: p.name)
    renamed = []
    for index, path in enumerate(files, start=1):
        if path.name.startswith("._"):
            continue
        if not _should_process_file(path):
            continue

        sequence = _infer_sequence_number_from_path(path, index)
        renamed.append(normalize_session_file(path, session_dir, sequence=sequence, date_str=date_str))
    return renamed


def _should_process_file(path: Path) -> bool:
    if path.name.endswith(".json") is False:
        return False
    if path.name.count("_") >= 1 and path.name.endswith(".json"):
        try:
            prefix = path.stem.split("_", 1)[0]
            int(prefix)
            return False
        except ValueError:
            return True
    return True


def _infer_sequence_number_from_path(path: Path, fallback: int) -> int:
    stem = path.stem
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
    else:
        prefix = stem
    try:
        return int(prefix)
    except ValueError:
        return fallback


def normalize_session_payload(payload: object) -> dict:
    if isinstance(payload, dict):
        normalized = dict(payload)
    else:
        normalized = {"version": 1, "events": []}
        if isinstance(payload, list):
            normalized["events"] = payload

    if "events" not in normalized:
        normalized["events"] = []
    if not isinstance(normalized["events"], list):
        normalized["events"] = list(normalized["events"])

    if "version" not in normalized:
        normalized["version"] = 1

    return normalized


def save_session_payload(path: Path, payload: object) -> Path:
    path = path.resolve()
    payload = normalize_session_payload(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def create_session_from_payload_text(payload_text: str, session_dir: Path, sequence: Optional[int] = None, date_str: Optional[str] = None) -> Path:
    session_dir = session_dir.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    if sequence is None:
        sequence = _infer_sequence_from_existing_files(session_dir)
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    target = session_dir / build_session_name(sequence, date_str)
    payload = json.loads(payload_text)
    save_session_payload(target, payload)
    return target


def read_payload_text(payload: Optional[str], from_stdin: bool = False, from_clipboard: bool = False) -> str:
    if payload is not None:
        return payload

    if from_stdin:
        return sys.stdin.read().strip()

    if from_clipboard:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            text = root.clipboard_get()
            root.destroy()
            return text.strip()
        except Exception:
            raise RuntimeError("Could not read clipboard content. Please make sure clipboard data is available.")

    raise ValueError("No payload source was provided.")


def read_payload_from_file(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def import_json_files_from_directory(source_dir: Path, session_dir: Path, date_str: Optional[str] = None) -> list[Path]:
    source_dir = source_dir.resolve()
    session_dir = session_dir.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    json_files = sorted(source_dir.glob("*.json"), key=lambda p: p.name)
    imported = []
    for index, path in enumerate(json_files, start=1):
        payload_text = path.read_text(encoding="utf-8-sig")
        sequence = _infer_sequence_from_existing_files(session_dir) + index - 1
        target = create_session_from_payload_text(payload_text, session_dir, sequence=sequence, date_str=date_str)
        imported.append(target)
    return imported


def normalize_session_file_contents(path: Path) -> Optional[Path]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"Skipping invalid JSON: {path.name}")
        return None

    if not _is_valid_session_payload(payload):
        payload = normalize_session_payload(payload)

    return save_session_payload(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize session JSON files or import one file or a directory of JSON payloads into the sessions folder.")
    parser.add_argument("source", nargs="?", help="Optional path to a single JSON file or a directory containing JSON files.")
    parser.add_argument("--payload", help="Optional raw JSON payload text to import directly.")
    parser.add_argument("--from-stdin", action="store_true", help="Read the JSON payload from standard input instead of a file.")
    parser.add_argument("--from-clipboard", action="store_true", help="Read the JSON payload from the system clipboard.")
    parser.add_argument("--sequence", type=int, default=None, help="Optional session sequence number for the imported payload.")
    parser.add_argument("--date", default=None, help="Optional date string for the imported payload, e.g. 20260719.")
    parser.add_argument("--sessions-dir", default="sessions", help="Directory containing session JSON files.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    sessions_dir = (project_root / args.sessions_dir).resolve()
    sessions_dir.mkdir(parents=True, exist_ok=True)

    if args.source:
        source_path = Path(args.source).resolve()
        if source_path.is_dir():
            imported = import_json_files_from_directory(source_path, sessions_dir, date_str=args.date)
            print("Imported session files:")
            for path in imported:
                print(path.name)
            return

        if source_path.is_file() and source_path.suffix.lower() == ".json":
            payload_text = source_path.read_text(encoding="utf-8-sig")
            output_path = create_session_from_payload_text(payload_text, sessions_dir, sequence=args.sequence, date_str=args.date)
            print(f"Imported session payload to {output_path.name}")
            return

        raise FileNotFoundError(f"Source path is not a valid JSON file or directory: {source_path}")

    if args.payload or args.from_stdin or args.from_clipboard:
        payload_text = read_payload_text(args.payload, from_stdin=args.from_stdin, from_clipboard=args.from_clipboard)
        output_path = create_session_from_payload_text(payload_text, sessions_dir, sequence=args.sequence, date_str=args.date)
        print(f"Imported session payload to {output_path.name}")
        return

    renamed = organize_sessions(sessions_dir)
    print("Renamed sessions:")
    for path in renamed:
        print(path.name)

    print("Normalized session payloads:")
    for path in sorted(sessions_dir.glob("*.json")):
        normalized_path = normalize_session_file_contents(path)
        if normalized_path is not None:
            print(normalized_path.name)


if __name__ == "__main__":
    main()
