from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

try:
    import gdown
except Exception as exc:
    raise ImportError(
        "This script requires gdown. Install it with:\n"
        "  pip install gdown"
    ) from exc


TRACE_DRIVE_FILE_ID = "1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"
EXPECTED_TASKS = [
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
]


def download_trace_zip(output_zip: Path, quiet: bool = False) -> Path:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={TRACE_DRIVE_FILE_ID}"
    gdown.download(url, str(output_zip), quiet=quiet, fuzzy=True)
    if not output_zip.exists():
        raise FileNotFoundError(f"Download appears to have failed: {output_zip}")
    return output_zip


def extract_zip(zip_path: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    return extract_root


def _looks_like_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "train.json").exists()
        and ((path / "eval.json").exists() or (path / "test.json").exists())
    )


def find_task_dirs(root: Path) -> Dict[str, Path]:
    matches: Dict[str, Path] = {}

    # Search a couple of levels deep because zip layouts vary.
    candidates = [root]
    candidates.extend([p for p in root.rglob("*") if p.is_dir()])

    for directory in candidates:
        if _looks_like_task_dir(directory):
            matches[directory.name] = directory

    return matches


def inspect_json_file(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return f"{path.name}: expected a list, got {type(data).__name__}"
        if len(data) == 0:
            return f"{path.name}: empty list"
        sample = data[0]
        if not isinstance(sample, dict):
            return f"{path.name}: first item is not a dict"
        if "prompt" not in sample or "answer" not in sample:
            return f"{path.name}: first item missing 'prompt'/'answer'"
        return None
    except Exception as exc:
        return f"{path.name}: failed to parse JSON ({exc})"


def validate_tasks(task_dirs: Dict[str, Path]) -> List[str]:
    issues: List[str] = []

    for task_name, task_dir in sorted(task_dirs.items()):
        train_path = task_dir / "train.json"
        eval_path = task_dir / "eval.json"
        test_path = task_dir / "test.json"

        if not train_path.exists():
            issues.append(f"{task_name}: missing train.json")
        if not eval_path.exists() and not test_path.exists():
            issues.append(f"{task_name}: missing both eval.json and test.json")

        for p in [train_path, eval_path, test_path]:
            if p.exists():
                maybe_issue = inspect_json_file(p)
                if maybe_issue is not None:
                    issues.append(f"{task_name}: {maybe_issue}")

    return issues


def print_summary(task_dirs: Dict[str, Path]) -> None:
    print("\nDiscovered TRACE task folders:")
    for task_name, task_dir in sorted(task_dirs.items()):
        print(f"  - {task_name}: {task_dir}")

    missing_expected = [t for t in EXPECTED_TASKS if t not in task_dirs]
    if missing_expected:
        print("\nExpected tasks not found:")
        for t in missing_expected:
            print(f"  - {t}")
    else:
        print("\nAll standard TRACE tasks were found.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="./TRACE_DATA")
    parser.add_argument("--zip-name", type=str, default="TRACE-Benchmark.zip")
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    zip_path = output_root / args.zip_name
    extract_root = output_root / "extracted"

    print(f"Downloading TRACE archive to: {zip_path}")
    download_trace_zip(zip_path, quiet=args.quiet)

    print(f"Extracting to: {extract_root}")
    extract_zip(zip_path, extract_root)

    task_dirs = find_task_dirs(extract_root)
    if not task_dirs:
        raise RuntimeError(
            "Download/extraction succeeded, but no TRACE task directories were found.\n"
            "Please inspect the extracted files manually."
        )

    print_summary(task_dirs)

    issues = validate_tasks(task_dirs)
    if issues:
        print("\nValidation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nValidation passed: task JSON files look compatible.")

    print("\nSuggested --trace-root values:")
    roots_to_try = sorted({str(p.parent) for p in task_dirs.values()})
    for root in roots_to_try:
        print(f"  - {root}")

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"\nRemoved zip file: {zip_path}")


if __name__ == "__main__":
    main()