#!/usr/bin/env python3
"""AI-generated script used to extract the saved notebook outputs into logs.

This reads saved stdout, including diagnostics and dataset previews.
It does not run training. Use --check to compare the existing logs without
changing them: python scripts/extract_logs.py --check
"""

import argparse
import json
from pathlib import Path
import sys


def text(value):
    """Read notebook text stored as a string or a list of fragments."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return "".join(value)
    raise ValueError("Expected notebook text to be a string or list of strings")


def extract(notebook_path, relative_path):
    notebook = json.loads(notebook_path.read_bytes())
    parts = [f"# Notebook: notebooks/{relative_path.as_posix()}\n"]
    saved_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        lines = text(cell.get("source", "")).splitlines()
        code = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        is_run = any(line.startswith("model_training_and_validation_with_mask(") for line in code)
        outputs = cell.get("outputs", [])
        if is_run and not outputs:
            raise ValueError(f"{relative_path}, cell {index}: training call has no saved output")
        if not outputs:
            continue
        chunks = []
        for output in outputs:
            if output.get("output_type") != "stream" or output.get("name") != "stdout":
                raise ValueError(f"{relative_path}, cell {index}: unsupported output; nothing exported")
            chunks.append(text(output["text"]))
        body = "".join(chunks)
        if not body:
            raise ValueError(f"{relative_path}, cell {index}: saved stdout is empty")
        parts.append(f"\n=== Cell {index} (zero-based) ===\n")
        if is_run:
            parts.append("Run settings from cell source:\n" + "\n".join(code) + "\n\n")
        parts.append(body)
        saved_cells += 1
    if not saved_cells:
        raise ValueError(f"{relative_path}: no saved outputs")
    return "".join(parts).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify logs without writing files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebooks_dir, logs_dir = root / "notebooks", root / "logs"
    try:
        notebooks = sorted(notebooks_dir.rglob("*.ipynb"))
        if not notebooks:
            raise ValueError(f"No notebooks found in {notebooks_dir}")
        # Validate every notebook before writing any logs.
        expected = {}
        for notebook in notebooks:
            relative = notebook.relative_to(notebooks_dir)
            expected[relative.with_suffix(".txt")] = extract(notebook, relative)
        existing = {path.relative_to(logs_dir) for path in logs_dir.rglob("*.txt")}
        extra = sorted(existing - expected.keys())
        if extra:
            raise ValueError("Unexpected log files: " + ", ".join(map(str, extra)))
        if args.check:
            missing = sorted(expected.keys() - existing)
            different = [path for path, data in expected.items()
                         if path in existing and (logs_dir / path).read_bytes() != data]
            if missing or different:
                raise ValueError(f"Missing logs: {list(map(str, missing))}; "
                                 f"different logs: {list(map(str, different))}")
        else:
            for path, data in expected.items():
                destination = logs_dir / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        print(f"{'Verified' if args.check else 'Extracted'} {len(expected)} notebook logs.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
