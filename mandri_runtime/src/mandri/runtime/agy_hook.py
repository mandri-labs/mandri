import json
import os
import sys

import httpx


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        event = sys.argv[1]
        response = httpx.post(
            os.environ["MANDRI_AGY_HOOK_URL"],
            headers={"Authorization": f"Bearer {os.environ['MANDRI_AGY_HOOK_TOKEN']}"},
            json={"event": event, "data": payload},
            timeout=float(os.environ.get("MANDRI_AGY_HOOK_TIMEOUT", "130")),
            trust_env=False,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            return 2
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, IndexError, httpx.HTTPError):
        print("Mandri hook unavailable", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
