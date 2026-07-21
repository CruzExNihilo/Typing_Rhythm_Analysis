from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-session and population analyses over all JSON files in the sessions folder.")
    parser.add_argument("--sessions-dir", default="sessions", help="Directory containing session JSON files.")
    parser.add_argument("--output-name", help="Optional output filename prefix for the single-session run.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for generated PDF files.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    sessions_dir = project_root / args.sessions_dir
    json_files = sorted(sessions_dir.glob("*.json"))

    if not json_files:
        print(f"No session JSON files found in {sessions_dir}.")
        return

    print(f"Batch processing {len(json_files)} session file(s)...")
    for session_file in json_files:
        print(f"- {session_file.name}")
        single_command = [sys.executable, str(project_root / "analysis_tools" / "single_session_analysis.py"), str(session_file)]
        if args.output_name:
            single_command.extend(["--output-name", f"{args.output_name}_{session_file.stem}"])
        if args.dpi:
            single_command.extend(["--dpi", str(args.dpi)])
        subprocess.run(single_command, cwd=project_root, check=True)

    population_command = [sys.executable, str(project_root / "analysis_tools" / "population_analysis.py")]
    if args.output_name:
        population_command.extend(["--output-name", args.output_name])
    if args.dpi:
        population_command.extend(["--dpi", str(args.dpi)])
    subprocess.run(population_command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
