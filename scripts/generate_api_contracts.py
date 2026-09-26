"""Generate versioned API contracts from the application and websocket registry."""

import argparse
import json
from pathlib import Path
from typing import Any

from mandri.api.app import create_app
from mandri.core.protocol.asyncapi import build_asyncapi

REPO_ROOT = Path(__file__).resolve().parents[1]


def generated_contracts() -> dict[Path, dict[str, Any]]:
    app = create_app()
    asyncapi = build_asyncapi(app.version)
    return {
        REPO_ROOT / "api-snapshots" / "openapi.json": app.openapi(),
        REPO_ROOT / "api-snapshots" / "asyncapi.json": asyncapi,
        REPO_ROOT / "mandri_core" / "tests" / "asyncapi_golden.json": asyncapi,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated contracts differ")
    args = parser.parse_args()
    changed: list[Path] = []
    for path, document in generated_contracts().items():
        text = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue
        changed.append(path)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        print(f"{'stale' if args.check else 'generated'}: {path.relative_to(REPO_ROOT)}")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
