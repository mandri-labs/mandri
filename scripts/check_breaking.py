"""Breaking-change gate for the OpenAPI snapshot.

Exit codes:
  0 - no breaking changes
  1 - breaking changes detected
  2 - oasdiff unavailable on PATH (see install hints printed by the script)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT = REPO_ROOT / "api-snapshots" / "openapi.json"
BASELINE = REPO_ROOT / "api-snapshots" / "baseline" / "openapi.json"

INSTALL_HINTS = (
    "oasdiff binary not found on PATH.",
    "oasdiff is a Go binary, not a Python package; `uv tool run` cannot fetch it.",
    "Install one of:",
    "  scoop install oasdiff",
    "  go install github.com/oasdiff/oasdiff@latest",
    "  download from https://github.com/oasdiff/oasdiff/releases",
)


def ensure_snapshot(path: Path, root_key: str) -> None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing snapshot: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid snapshot: {path}: {exc}") from exc
    if root_key not in document or "info" not in document:
        raise SystemExit(f"snapshot missing '{root_key}' or 'info': {path}")


def update_baseline() -> None:
    ensure_snapshot(CURRENT, "openapi")
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(CURRENT.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"baseline updated: {BASELINE}")


def run_gate() -> int:
    ensure_snapshot(CURRENT, "openapi")
    ensure_snapshot(BASELINE, "openapi")
    oasdiff = shutil.which("oasdiff")
    if oasdiff is None:
        for line in INSTALL_HINTS:
            print(line, file=sys.stderr)
        return 2
    result = subprocess.run(
        [oasdiff, "breaking", str(BASELINE), str(CURRENT)],
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if result.returncode == 0:
        print("no breaking changes")
    else:
        print("breaking changes detected", file=sys.stderr)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate the OpenAPI snapshot against its baseline with oasdiff."
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="copy the current OpenAPI snapshot over the baseline",
    )
    args = parser.parse_args()
    if args.update_baseline:
        update_baseline()
        return 0
    return run_gate()


if __name__ == "__main__":
    raise SystemExit(main())
