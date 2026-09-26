import json

from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.launch import HarnessLaunch


def native_launch(
    harness: HarnessKind,
    model: str = "default",
    effort: str | None = None,
    resume_id: HarnessSessionId | None = None,
) -> HarnessLaunch:
    args: list[str] = []
    if harness is HarnessKind.PI:
        if model != "default":
            args.extend(("--model", model))
        if effort:
            args.extend(("--thinking", effort))
        if resume_id:
            args.extend(("--session", str(resume_id)))
        return HarnessLaunch(
            {},
            strip=("MANDRI_API_KEY", "MANDRI_PI_BASE_URL", "MANDRI_PI_MODEL"),
            args=tuple(args),
        )
    if harness is HarnessKind.CODEX:
        args.extend(("-c", 'model_provider="openai"', "-c", 'forced_login_method="chatgpt"'))
        if model != "default":
            args.extend(("-c", f"model={json.dumps(model)}"))
        if effort:
            args.extend(("-c", f"model_reasoning_effort={json.dumps(effort)}"))
        return HarnessLaunch(
            {},
            strip=("MANDRI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"),
            args=tuple(args),
        )
    if harness is HarnessKind.CLAUDE:
        args.extend(("--model", model))
        if effort:
            args.extend(("--effort", effort))
        if resume_id:
            args.extend(("--resume", str(resume_id)))
        return HarnessLaunch(
            {},
            strip=(
                "MANDRI_API_KEY",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_MODEL",
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX",
                "CLAUDE_CODE_USE_FOUNDRY",
                "CLAUDE_CODE_USE_ANTHROPIC_AWS",
            ),
            args=tuple(args),
        )
    if harness is HarnessKind.AGY:
        if model != "default":
            args.extend(("--model", model))
        if effort:
            args.extend(("--effort", effort))
        if resume_id:
            args.extend(("--conversation", str(resume_id)))
        return HarnessLaunch(
            {"AGY_CLI_DISABLE_AUTO_UPDATE": "true"},
            strip=(
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "GOOGLE_GEMINI_BASE_URL",
                "GOOGLE_GENAI_USE_VERTEXAI",
            ),
            args=tuple(args),
        )
    raise ValueError("Native models require Codex, Claude Code, Antigravity or Pi")
