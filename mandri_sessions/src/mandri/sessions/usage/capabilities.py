def native_usage_capabilities(harness: str) -> dict[str, object]:
    supported = harness in {"codex", "claude", "agy", "pi"}
    history = harness in {"codex", "claude", "pi"}
    return {
        "harness": harness,
        "live": "available" if supported else "unsupported",
        "history": "available" if history else "unsupported",
        "history_parser_version": 2 if history else None,
        "history_granularity": "request" if history else None,
        "account_push": "available" if harness in {"codex", "claude"} else "unsupported",
        "account_read": "requires_qualification" if harness in {"codex", "agy"} else "unsupported",
        "model_attribution": "partial" if supported else "unsupported",
        "runtime_qualified": False,
        "history_limitations": {
            "codex": "host_jsonl;fork_prefix_requires_ordinal;missing_model_or_usage_is_a_gap",
            "claude": "host_jsonl;assistant_message_usage_only;fork_ownership_requires_evidence",
            "pi": (
                "host_jsonl;unattributed_auxiliary_usage_is_partial;"
                "fork_ownership_requires_evidence"
            ),
        }.get(harness, "historical_reader_unavailable"),
        "quota_limitations": "agy_standalone_payload_unqualified;claude_headless_poll_unsupported",
    }
