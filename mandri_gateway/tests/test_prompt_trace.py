import logging

from mandri.gateway.prompt_trace import PromptFingerprint, PromptTrace


def test_envelope_changes_do_not_change_prompt_fingerprints():
    body = {"instructions": "fixture", "input": [{"role": "user", "content": "hello"}], "tools": []}
    changed = {**body, "client_metadata": {"request_id": "new"}, "metadata": {"timestamp": "new"}}
    assert PromptFingerprint.from_body(body) == PromptFingerprint.from_body(changed)


def test_followup_preserves_the_system_and_reports_the_retained_prefix(caplog):
    caplog.set_level(logging.DEBUG, logger="mandri.gateway.prompt_trace")
    trace = PromptTrace()
    body = {
        "instructions": "private instruction",
        "input": [{"role": "user", "content": "private user text"}],
        "tools": [],
    }
    trace.observe("fixture", body)
    trace.observe(
        "fixture",
        {
            **body,
            "input": [
                *body["input"],
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "followup"},
            ],
        },
    )
    assert "instructions_changed=False" in caplog.text
    assert "common_input_items=1 input_items=3" in caplog.text
    assert "private instruction" not in caplog.text
    assert "private user text" not in caplog.text
