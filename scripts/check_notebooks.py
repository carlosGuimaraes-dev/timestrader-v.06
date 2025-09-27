#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def check_notebook(path: Path) -> int:
    nb = json.loads(path.read_text())
    errors = 0
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        try:
            compile(src, f"{path.name}:cell{i}", "exec")
        except SyntaxError as e:
            print(f"SyntaxError in {path}:{i}: {e}")
            errors += 1
    return errors


def main() -> int:
    nb_paths = list(Path(".").rglob("*.ipynb"))
    if not nb_paths:
        print("No notebooks found.")
        return 0
    total_errors = 0
    for p in nb_paths:
        total_errors += check_notebook(p)
    if total_errors:
        print(f"Found {total_errors} syntax errors in notebooks.")
        return 1
    print("All notebooks parsed and compiled successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
