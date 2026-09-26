import json
import re
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

READ_TOOLS = frozenset({"view_file", "list_dir", "find_by_name", "grep_search"})
EDIT_TOOLS = frozenset(
    {"write_to_file", "replace_file_content", "multi_replace_file_content", "notebook_edit"}
)
MODES = frozenset({"default", "acceptEdits", "plan", "bypassPermissions"})


def tool_key(name: str, args: dict[str, Any]) -> str:
    relevant = {k: v for k, v in args.items() if k not in {"toolAction", "toolSummary"}}
    return name + ":" + json.dumps(relevant, sort_keys=True)


class AgyPolicy:
    def __init__(
        self,
        mode: str,
        cwd: Path,
        permissions: dict[str, Any],
        runtime_root: Path | None = None,
    ) -> None:
        self.mode = mode
        self.cwd = cwd.resolve()
        self.permissions = permissions
        self.grants: set[str] = set()
        self.runtime_root = runtime_root

    def decision(self, name: str, args: dict[str, Any]) -> str:
        if name == "ask_question":
            return "ask"
        if self.mode == "plan" and name not in READ_TOOLS:
            return "deny"
        resource = self.resource(name, args)
        for decision in ("deny", "ask", "allow"):
            rules = self.permissions.get(decision, [])
            if isinstance(rules, list) and any(
                self.matches(rule, resource, decision) for rule in rules
            ):
                return decision
        if self.mode == "bypassPermissions" or tool_key(name, args) in self.grants:
            return "allow"
        if (
            (name in READ_TOOLS or (self.mode == "acceptEdits" and name in EDIT_TOOLS))
            and resource is not None
            and resource[0] in {"read_file", "write_file"}
            and self.path(resource[1]).is_relative_to(self.cwd)
        ):
            return "allow"
        return "ask"

    def path(self, value: str) -> Path:
        path = Path(value)
        if self.runtime_root is not None and path.is_relative_to(self.runtime_root):
            path = self.cwd / path.relative_to(self.runtime_root)
        return (path if path.is_absolute() else self.cwd / path).resolve()

    def resource(self, name: str, args: dict[str, Any]) -> tuple[str, str] | None:
        if name == "run_command":
            return ("command", str(args.get("CommandLine", "")))
        if name in READ_TOOLS | EDIT_TOOLS:
            for key in (
                "AbsolutePath",
                "TargetFile",
                "DirectoryPath",
                "SearchDirectory",
                "SearchPath",
            ):
                value = args.get(key)
                if isinstance(value, str):
                    return ("write_file" if name in EDIT_TOOLS else "read_file", value)
        if name in {"read_url_content", "open_browser_url"}:
            return ("read_url", str(args.get("Url", args.get("URL", ""))))
        return None

    def matches(
        self, rule: object, resource: tuple[str, str] | None, decision: str = "allow"
    ) -> bool:
        if rule == "*":
            return True
        if not isinstance(rule, str) or resource is None:
            return False
        action, target = resource
        prefix = action + "("
        if not rule.startswith(prefix) or not rule.endswith(")"):
            return False
        pattern = rule[len(prefix) : -1]
        if pattern == "*":
            return True
        if action in {"read_file", "write_file"}:
            return self.path(target).is_relative_to(self.path(pattern))
        if action == "read_url":
            host = urlparse(target).hostname or target
            return host == pattern or host.endswith("." + pattern)
        if re.search(r"[;&|$><`\r\n]", target):
            return decision == "deny" or target == pattern
        if pattern.startswith("regex:"):
            try:
                return re.search(pattern[6:], target) is not None
            except re.error:
                return False
        try:
            expected, actual = shlex.split(pattern), shlex.split(target)
            return bool(expected) and actual[: len(expected)] == expected
        except ValueError:
            return target == pattern
