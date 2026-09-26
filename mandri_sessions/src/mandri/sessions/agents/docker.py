import dataclasses
import os
from pathlib import Path

from mandri.core.ids import HarnessKind
from mandri.core.ports.agents import AgentDiscovery, AgentDiscoveryResult
from mandri.core.types.agents import Agent
from mandri.core.types.execution import ProtectionError
from mandri.core.types.sessions import Session
from mandri.sessions.agents.agy import AgyAgentDiscovery
from mandri.sessions.agents.claude import ClaudeAgentDiscovery
from mandri.sessions.agents.codex import CodexAgentDiscovery
from mandri.sessions.agents.docker_agy import check_discovery_state
from mandri.sessions.agents.opencode import OpencodeAgentDiscovery
from mandri.sessions.execution_context import DockerSessionContext


def _check_tree(context: DockerSessionContext, root: Path) -> None:
    if not root.exists():
        if root.is_symlink():
            context.state_path(root)
        return
    root = context.state_path(root)
    total = 0
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            total += 1
            path = Path(directory) / name
            if total > 20000 or path.is_symlink():
                raise ProtectionError(
                    "native_state_incompatible", "Native discovery state is unsupported"
                )
            context.state_path(path)


def discover_docker_agents(session: Session) -> AgentDiscoveryResult:
    if session.native_id is None or session.execution_context is None:
        return AgentDiscoveryResult(session.harness, [], frozenset(), complete=False)
    context = DockerSessionContext.from_session(session)
    if session.harness is HarnessKind.PI:
        return AgentDiscoveryResult(session.harness, [], frozenset(), complete=False)
    discovery: AgentDiscovery
    if session.harness is HarnessKind.CODEX:
        home = context.state_root / ".codex"
        _check_tree(context, home / "sessions")
        if (home / "state_5.sqlite").exists() or (home / "state_5.sqlite").is_symlink():
            context.state_path(home / "state_5.sqlite")
        discovery = CodexAgentDiscovery(home)
    elif session.harness is HarnessKind.CLAUDE:
        home = context.state_root / ".claude/projects"
        _check_tree(context, home)
        discovery = ClaudeAgentDiscovery(home)
    elif session.harness is HarnessKind.AGY:
        home = context.state_root / ".gemini"
        check_discovery_state(context, home)
        discovery = AgyAgentDiscovery(home)
    else:
        database = context.state_root / ".local/share/opencode/opencode.db"
        if database.exists() or database.is_symlink():
            database = context.state_path(database)
        discovery = OpencodeAgentDiscovery(database)
    result = discovery.discover([session])
    agents = [
        dataclasses.replace(
            agent,
            transcript_path=str(context.state_path(Path(agent.transcript_path)))
            if agent.transcript_path
            else None,
        )
        for agent in result.agents
    ]
    return dataclasses.replace(result, agents=agents)


def contextual_agent(agent: Agent, parent: Session) -> Agent:
    if agent.harness is not parent.harness:
        raise ProtectionError("native_state_incompatible", "Agent execution context differs")
    context = DockerSessionContext.from_session(parent)
    if agent.transcript_path is None:
        return agent
    return dataclasses.replace(
        agent, transcript_path=str(context.state_path(Path(agent.transcript_path)))
    )
