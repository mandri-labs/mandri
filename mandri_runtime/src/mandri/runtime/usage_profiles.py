import hashlib
import os
from collections.abc import Mapping
from pathlib import Path


def usage_profile_id(
    harness: str,
    env: Mapping[str, str] | None,
    *,
    isolated_scope: str | None = None,
    agy_root: Path | None = None,
) -> str:
    environment = os.environ if env is None else env
    home = Path(environment.get("HOME") or environment.get("USERPROFILE") or str(Path.home()))
    if isolated_scope is not None:
        identity = f"isolated:{isolated_scope}"
    elif harness == "codex":
        identity = str(Path(environment.get("CODEX_HOME") or home / ".codex").absolute())
    elif harness == "claude":
        identity = str(Path(environment.get("CLAUDE_CONFIG_DIR") or home / ".claude").absolute())
    elif harness == "pi":
        identity = str(
            Path(environment.get("PI_CODING_AGENT_DIR") or home / ".pi" / "agent").absolute()
        )
    else:
        identity = str((agy_root or home / ".gemini").absolute())
    return f"profile:{harness}:" + hashlib.sha256(identity.encode()).hexdigest()
