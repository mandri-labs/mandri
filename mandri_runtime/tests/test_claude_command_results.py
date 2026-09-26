from mandri.runtime.control.claude_command_results import command_result


def test_context_uses_native_kinds_without_clamping_or_recounting_deferred_tokens() -> None:
    result = command_result(
        {
            "context_usage": {
                "model": "example",
                "total_tokens": 210000,
                "raw_max_tokens": 200000,
                "percentage": 105,
                "over_limit": {"tokens_over": 10000, "kind": "compaction_window"},
                "categories": [
                    {"name": "Tools", "tokens": 2000, "kind": "deferred"},
                    {"name": "Messages", "tokens": 208000, "kind": "used"},
                    {"name": "Available", "tokens": 0, "kind": "free"},
                ],
                "mcp_tools": [{"name": "read", "server_name": "workspace", "tokens": 200}],
                "memory_files": [{"path": "AGENTS.md", "type": "Project", "tokens": 100}],
                "agents": [{"agent_type": "reviewer", "source": "projectSettings", "tokens": 100}],
                "skills": [
                    {"name": "audit", "source": "plugin", "plugin_name": "example", "tokens": 120}
                ],
            }
        }
    )
    assert result is not None
    fields = {row["label"]: row["value"] for row in result["fields"]}
    assert fields["Window used (%)"] == 105
    assert fields["Tokens over compaction window"] == 10000
    assert fields["Tools · Outside window (tokens)"] == 2000
    assert fields["Available · Available (tokens)"] == 0
    assert fields["Skill · audit · plugin · example (tokens)"] == 120
    assert fields["Tokens in use"] == 210000


def test_usage_preserves_server_meter_order_and_extra_usage_minor_units() -> None:
    result = command_result(
        {
            "usage_report": {
                "session": {
                    "total_cost_usd": 0.25,
                    "total_api_duration_ms": 500,
                    "total_duration_ms": 1000,
                    "total_lines_added": 3,
                    "total_lines_removed": 1,
                    "model_usage": {
                        "example": {
                            "inputTokens": 10,
                            "outputTokens": 20,
                            "thinkingTokens": 5,
                            "costUSD": 0.25,
                        }
                    },
                },
                "rate_limits": {
                    "limits": [
                        {
                            "kind": "new_meter",
                            "group": "weekly",
                            "percent": 45,
                            "resets_at": None,
                            "severity": "warning",
                            "is_active": True,
                            "scope": {"surface": {"display_name": "Example surface"}},
                        },
                        {
                            "kind": "session",
                            "group": "session",
                            "percent": 10,
                            "resets_at": "2026-09-24T18:00:00Z",
                        },
                    ],
                    "extra_usage": {
                        "is_enabled": True,
                        "monthly_limit": 10000,
                        "used_credits": 125,
                        "utilization": 1.25,
                        "currency": "USD",
                    },
                },
            }
        }
    )
    assert result is not None
    fields = {row["label"]: row["value"] for row in result["fields"]}
    assert fields["Extra usage spent (USD minor units)"] == 125
    assert fields["example · Thinking tokens (included in output)"] == 5
    assert fields["weekly · new_meter · Example surface · Current meter"] is True
    labels = list(fields)
    assert labels.index("weekly · new_meter · Example surface · Used (%)") < labels.index(
        "session · session · Used (%)"
    )


def test_missing_plan_usage_is_unavailable_not_zero() -> None:
    result = command_result(
        {"usage_report": {"session": {"total_cost_usd": 0}, "rate_limits": None}}
    )
    assert result is not None
    assert result["fields"][-1] == {"label": "Plan usage", "value": "Unavailable for this session"}


def test_unknown_or_invalid_shapes_leave_canonical_assistant_text_available() -> None:
    assert command_result({"message": {"content": [{"type": "text", "text": "Report"}]}}) is None
    assert command_result({"context_usage": {"total_tokens": True}}) is None
    assert command_result({"usage_report": {"session": {"total_cost_usd": float("nan")}}}) is None
