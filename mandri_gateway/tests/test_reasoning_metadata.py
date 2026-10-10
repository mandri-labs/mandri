import pytest
from mandri.gateway.reasoning_metadata import (
    ReasoningInfo,
    parse_reasoning_entry,
    parse_template_reasoning,
)


@pytest.mark.parametrize(
    "entry",
    [
        {"reasoning": {"supported_efforts": ["low", "high"], "default": "high"}},
        {"capabilities": {"reasoning": {"allowed_options": ["low", "high"], "default": "high"}}},
        {"reasoning_efforts": ["low", "high"], "default_effort": "high"},
    ],
)
def test_explicit_model_options(entry):
    assert parse_reasoning_entry(entry) == ReasoningInfo(["low", "high"], "high")


@pytest.mark.parametrize(
    "entry",
    [None, {}, {"reasoning": True}, {"reasoning_efforts": []}, {"reasoning_efforts": ["low", 2]}],
)
def test_missing_or_malformed_options_are_unknown(entry):
    assert parse_reasoning_entry(entry) is None


def test_explicitly_disabled_reasoning_is_authoritative():
    assert parse_reasoning_entry({"capabilities": {"reasoning": False}}) == ReasoningInfo([])


def test_template_validation_declares_levels_and_default_without_execution():
    template = """
    {% set resolved = reasoning_effort|default('high') %}
    {% if resolved not in ('low', 'high') %}
      {{ raise_exception('Unsupported effort') }}
    {% endif %}
    {{ dangerous_function() }}
    """
    assert parse_template_reasoning(template) == ReasoningInfo(["low", "high"], "high")


@pytest.mark.parametrize(
    "template",
    [
        None,
        "{% broken %}",
        "{% if reasoning_effort not in ('low', 'high') %}optional text{% endif %}",
        "{% if other_option not in ('low', 'high') %}{{ raise_exception('no') }}{% endif %}",
        "{% if reasoning_effort not in dynamic_values %}{{ raise_exception('no') }}{% endif %}",
        "{% if reasoning_effort not in ('low', 3) %}{{ raise_exception('no') }}{% endif %}",
    ],
)
def test_template_without_literal_effort_validation_is_unknown(template):
    assert parse_template_reasoning(template) is None
