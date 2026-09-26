import pytest
from mandri.sessions.usage.account_queries import agy_usage_snapshot


@pytest.mark.parametrize("field", ["plan_name", "tier_display_name", "plan_tier"])
def test_agy_reads_subscription_and_distinguishes_group_windows(field):
    account = agy_usage_snapshot(
        "profile",
        {
            "command": {
                "data": {
                    field: "Google AI Pro",
                    "groups": [
                        {
                            "name": "Gemini Models",
                            "buckets": [
                                {"id": "gemini-5h", "window": "5h", "remaining_fraction": 1},
                                {
                                    "id": "gemini-weekly",
                                    "window": "weekly",
                                    "remaining_fraction": 0,
                                },
                                {"id": "disabled", "window": "weekly"},
                            ],
                        }
                    ],
                }
            }
        },
        observed_at_ms=1,
    )
    assert account.plan == "Google AI Pro"
    assert account.status == "available"
    assert [window["remaining_fraction"] for window in account.windows] == [1, 0]
    assert [window["bucket_id"] for window in account.windows] == ["gemini-5h", "gemini-weekly"]
    assert [window["window_duration_minutes"] for window in account.windows] == [300, 10080]


def test_agy_preserves_subscription_without_inventing_quota_values():
    account = agy_usage_snapshot(
        "profile",
        {"command": {"data": {"plan_name": "Google AI Ultra", "groups": []}}},
        observed_at_ms=1,
    )
    assert account.plan == "Google AI Ultra"
    assert account.has_account_data
    assert account.status == "unavailable"
    assert not account.windows
