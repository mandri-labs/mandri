import dataclasses


@dataclasses.dataclass(frozen=True)
class PromptAttachment:
    path: str
    media_type: str
    data: bytes


@dataclasses.dataclass(frozen=True)
class UserPrompt:
    text: str
    attachments: tuple[PromptAttachment, ...] = ()


def prompt_text(content: str | UserPrompt) -> str:
    return content if isinstance(content, str) else content.text
