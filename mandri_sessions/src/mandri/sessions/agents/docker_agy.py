from pathlib import Path

from mandri.core.types.execution import ProtectionError
from mandri.sessions.execution_context import DockerSessionContext


def _optional(context: DockerSessionContext, path: Path, *, directory: bool = False) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    resolved = context.state_path(path)
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise ProtectionError("native_state_incompatible", "Native discovery state is unsupported")
    return True


def check_discovery_state(context: DockerSessionContext, root: Path) -> None:
    if not _optional(context, root, directory=True):
        return
    for relative in (
        "mandri-session.json",
        "antigravity-cli/conversation_summaries.db",
        "antigravity-cli/conversation_summaries.db-wal",
        "antigravity-cli/conversation_summaries.db-shm",
        "antigravity-cli/conversation_summaries.db-journal",
        "antigravity-cli/cache/conversation_metadata.json",
        "antigravity-cli/cache/last_conversations.json",
    ):
        _optional(context, root / relative)
    bindings = root / "bindings"
    if _optional(context, bindings, directory=True):
        for count, path in enumerate(bindings.iterdir(), 1):
            _limit(count)
            if path.suffix == ".json":
                _optional(context, path)
    brain = root / "antigravity-cli/brain"
    if not _optional(context, brain, directory=True):
        return
    for count, conversation in enumerate(brain.iterdir(), 1):
        _limit(count)
        if not conversation.is_dir() and not conversation.is_symlink():
            continue
        _optional(context, conversation, directory=True)
        for name in ("transcript_full.jsonl", "transcript.jsonl"):
            _optional(context, conversation / ".system_generated/logs" / name)


def _limit(count: int) -> None:
    if count > 20000:
        raise ProtectionError("native_state_incompatible", "Native discovery state is unsupported")
