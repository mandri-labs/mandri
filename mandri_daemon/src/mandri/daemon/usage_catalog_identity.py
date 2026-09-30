import hashlib
import re
from dataclasses import replace

from mandri.core.types.usage import UsagePrice

NATIVE_PROVIDERS = {"openai", "anthropic", "gemini"}
NATIVE_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "thinking")


def model_aliases(model: str, canonical: object = None) -> set[str]:
    aliases = {model}
    if isinstance(canonical, str) and canonical and ":" not in model:
        aliases.add(canonical)
        aliases.add(canonical.partition("/")[2] or canonical)
    for identifier in tuple(aliases):
        plain = identifier.partition("/")[2] or identifier
        if plain.startswith("claude-"):
            normalized = re.sub(r"(?<=\d)\.(?=\d)", "-", plain)
            reordered = re.sub(
                r"^claude-(\d+(?:-\d+)?)-(haiku|sonnet|opus)(.*)$",
                r"claude-\2-\1\3",
                normalized,
            )
            aliases.update((normalized, reordered))
    return aliases


def native_aliases(aliases: set[str]) -> set[str]:
    result = set(aliases)
    for alias in aliases:
        plain = alias.partition("/")[2] or alias
        result.add(plain)
        if plain.startswith("gemini-"):
            result.add("models/" + plain)
        if plain.startswith(("gemini-", "claude-", "gpt-oss-")):
            base = plain.removesuffix("-preview")
            result.update(f"{base}-{effort}" for effort in NATIVE_EFFORTS)
    return result


def project_prices(
    prices: list[UsagePrice],
    aliases: set[str],
    provider: str,
    source_provider: str,
    source_model: str,
) -> list[UsagePrice]:
    targets = {provider: aliases}
    if provider in NATIVE_PROVIDERS:
        targets["native:agy"] = native_aliases(aliases)
        targets["native:pi"] = aliases
    result = []
    for target, identifiers in targets.items():
        for alias in sorted(identifiers):
            for price in prices:
                identity = hashlib.sha256(f"{price.price_id}:{target}:{alias}".encode()).hexdigest()
                result.append(
                    replace(
                        price,
                        price_id="current:" + identity,
                        provider=target,
                        model=alias,
                        source_provider=source_provider,
                        source_model=source_model,
                    )
                )
    return result
