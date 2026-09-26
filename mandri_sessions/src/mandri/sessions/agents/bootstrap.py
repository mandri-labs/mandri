from pathlib import Path

from mandri.core.ports.database import DatabasePort
from mandri.core.types.config import SessionsConfig
from mandri.sessions.agents.agy import AgyAgentDiscovery
from mandri.sessions.agents.claude import ClaudeAgentDiscovery
from mandri.sessions.agents.codex import CodexAgentDiscovery
from mandri.sessions.agents.opencode import OpencodeAgentDiscovery
from mandri.sessions.agents.service import AgentHistory
from mandri.sessions.agents.store import AgentStore
from mandri.sessions.agy_profiles import default_agy_root
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.resolver import TranscriptResolver


def build_agent_history(
    db: DatabasePort,
    sessions: SessionsService,
    config: SessionsConfig,
    transcripts: TranscriptResolver,
    default_opencode_db: Path,
) -> AgentHistory:
    codex_home = Path(config.codex_home) if config.codex_home else Path.home() / ".codex"
    claude_home = (
        Path(config.claude_config_dir) if config.claude_config_dir else Path.home() / ".claude"
    )
    opencode_db = Path(config.opencode_db_path) if config.opencode_db_path else default_opencode_db
    return AgentHistory(
        AgentStore(db),
        sessions,
        transcripts,
        [
            CodexAgentDiscovery(codex_home),
            ClaudeAgentDiscovery(claude_home / "projects"),
            OpencodeAgentDiscovery(opencode_db),
            AgyAgentDiscovery(
                Path(config.agy_home) if config.agy_home else default_agy_root(),
                Path(config.agy_profiles_dir) if config.agy_profiles_dir else None,
            ),
        ],
    )
