import re
from collections.abc import Callable, Iterable

from mandri.gateway.surrogate.allocation import unique_alias
from mandri.gateway.surrogate.formats import SurrogateGenerator, valid
from mandri.gateway.surrogate.types import Mapping, PathRoot


class PathAliases:
    def __init__(self, roots: Iterable[PathRoot]) -> None:
        self.components: dict[str, str] = {}
        for root in roots:
            self.remember(root)

    def remember(self, root: PathRoot) -> None:
        originals = re.split(r"[/\\]", root.original)
        surrogates = re.split(r"[/\\]", root.surrogate)
        if len(originals) == len(surrogates):
            for original, surrogate in zip(originals, surrogates, strict=True):
                if original and original != surrogate:
                    self.components.setdefault(original, surrogate)

    def generate(
        self,
        original: str,
        mappings: dict[str, Mapping],
        generator: SurrogateGenerator,
        available: Callable[[str, str], bool],
    ) -> str:
        parts = re.split(r"([/\\])", original)
        selected = [
            index
            for index, part in enumerate(parts)
            if part not in {"", "/", "\\", "home", "Users", "workspace"}
            and not re.fullmatch(r"[a-zA-Z]:", part)
        ]
        if not selected:
            selected = [
                next(
                    index
                    for index in range(len(parts) - 1, -1, -1)
                    if parts[index] not in {"", "/", "\\"}
                )
            ]
        for index in selected:
            part = parts[index]
            known = mappings.get(part)
            if known is not None and valid("username", known.surrogate, part):
                parts[index] = known.surrogate
                continue
            if part not in self.components:
                alias = generator.generate("username", part)
                self.components[part] = unique_alias(
                    part,
                    alias,
                    lambda original, candidate: (
                        available(original, candidate) and candidate not in self.components.values()
                    ),
                )
            parts[index] = self.components[part]
        return "".join(parts)
