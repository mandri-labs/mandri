import dataclasses
from collections.abc import Callable, Sequence

from mandri.core.types.conversation_status import ConversationStatus, WorkObservation, WorkOutcome


@dataclasses.dataclass
class ObservedWork:
    armed: bool = False
    cycle_key: str | None = None
    terminal_key: str | None = None
    terminal_content_key: str | None = None
    terminal_outcome: WorkOutcome | None = None


def reduce_observations(
    current: ConversationStatus,
    work: ObservedWork,
    observations: Sequence[WorkObservation],
    claim_completion: Callable[[str], bool],
) -> ConversationStatus:
    values = current.model_dump()
    for observation in observations:
        if observation.source is not None:
            values["observation_source"] = observation.source
        if observation.progress and not work.armed:
            work.armed = True
            work.cycle_key = observation.key
            work.terminal_key = None
            work.terminal_outcome = None
            work.terminal_content_key = None
        if observation.outcome is not None and observation.key is not None and work.armed:
            work.terminal_key = observation.key
            work.terminal_outcome = observation.outcome
            work.terminal_content_key = observation.content_key
        if observation.state is not None:
            values["work_state"] = observation.state
        if (
            observation.state == "idle"
            and work.armed
            and work.terminal_key is not None
            and work.terminal_outcome is not None
        ):
            receipt = work.terminal_content_key or f"{work.cycle_key}:{work.terminal_key}"
            if claim_completion(receipt):
                values["completion_revision"] += 1
                values["completion_key"] = receipt
                values["completion_content_key"] = work.terminal_content_key
                values["outcome"] = work.terminal_outcome
            work.armed = False
            work.cycle_key = None
            work.terminal_key = None
            work.terminal_outcome = None
            work.terminal_content_key = None
    values["cycle_active"] = work.armed
    values["pending_outcome"] = work.terminal_outcome
    candidate = ConversationStatus(**values)
    if candidate != current:
        candidate.revision += 1
    return candidate
