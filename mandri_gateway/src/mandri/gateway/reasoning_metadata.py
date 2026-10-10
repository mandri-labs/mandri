from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, TemplateSyntaxError, nodes


@dataclass(frozen=True)
class ReasoningInfo:
    efforts: list[str]
    default_effort: str | None = None


def reasoning_info(values: Any, default: Any = None) -> ReasoningInfo | None:
    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(value, str) and value for value in values):
        return None
    efforts = list(dict.fromkeys(values))
    return ReasoningInfo(
        efforts, default if isinstance(default, str) and default in efforts else None
    )


def parse_reasoning_entry(entry: Any) -> ReasoningInfo | None:
    if not isinstance(entry, dict):
        return None
    capabilities = entry.get("capabilities")
    if entry.get("reasoning") is False or (
        isinstance(capabilities, dict) and capabilities.get("reasoning") is False
    ):
        return ReasoningInfo([])
    sources = [entry.get("reasoning")]
    if isinstance(capabilities, dict):
        sources.append(capabilities.get("reasoning"))
    for source in sources:
        if isinstance(source, dict):
            for key in ("supported_efforts", "allowed_options"):
                info = reasoning_info(
                    source.get(key), source.get("default", source.get("default_effort"))
                )
                if info is not None:
                    return info
    info = reasoning_info(entry.get("reasoning_efforts"), entry.get("default_effort"))
    if info is not None:
        return info
    options = entry.get("reasoning_options")
    options = [options] if isinstance(options, dict) else options
    if isinstance(options, list):
        values = []
        for option in options:
            if isinstance(option, dict) and option.get("type") == "effort":
                info = reasoning_info(option.get("values"))
                if info is not None:
                    values.extend(info.efforts)
        return reasoning_info(values)
    return None


def parse_template_reasoning(template: Any) -> ReasoningInfo | None:
    if not isinstance(template, str) or len(template) > 100_000:
        return None
    try:
        tree = Environment().parse(template)
        return _template_reasoning(tree)
    except (TemplateSyntaxError, RecursionError):
        return None


def _template_reasoning(tree: nodes.Template) -> ReasoningInfo | None:
    names = {"reasoning_effort"}
    default = None
    for assignment in tree.find_all(nodes.Assign):
        value = assignment.node
        if (
            isinstance(assignment.target, nodes.Name)
            and isinstance(value, nodes.Filter)
            and value.name == "default"
            and isinstance(value.node, nodes.Name)
            and value.node.name in names
        ):
            names.add(assignment.target.name)
            if value.args and isinstance(value.args[0], nodes.Const):
                default = value.args[0].value
    candidates = []
    for condition in tree.find_all(nodes.If):
        test = condition.test
        if not (
            isinstance(test, nodes.Compare)
            and isinstance(test.expr, nodes.Name)
            and test.expr.name in names
            and len(test.ops) == 1
            and test.ops[0].op == "notin"
            and isinstance(test.ops[0].expr, (nodes.Tuple, nodes.List))
        ):
            continue
        if not any(
            isinstance(call.node, nodes.Name) and call.node.name == "raise_exception"
            for child in condition.body
            for call in child.find_all(nodes.Call)
        ):
            continue
        items = test.ops[0].expr.items
        if all(isinstance(item, nodes.Const) for item in items):
            info = reasoning_info(
                [item.value for item in items if isinstance(item, nodes.Const)], default
            )
            if info is not None:
                candidates.append(info)
    if candidates and all(info == candidates[0] for info in candidates):
        return candidates[0]
    return None
