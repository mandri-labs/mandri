from decimal import Decimal, localcontext

from mandri.core.types.usage import UsagePrice

OPENCODE_GO_SOURCE = "https://opencode.ai/docs/go/#usage-limits"
OPENCODE_GO_SCHEDULED_MODELS = {
    "deepseek-v4-flash": ("0.15", "0.003", "0.6"),
    "deepseek-v4.1-flash": ("0.15", "0.003", "0.6"),
    "deepseek-v4-flash-vision-exp": ("0.15", "0.003", "0.6"),
    "deepseek-v4-pro": ("0.66", "0.022", "1.98"),
}


def opencode_go_prices(reviewed_at: int) -> tuple[UsagePrice, ...]:
    peak = [
        [day * 1440 + start, day * 1440 + end]
        for day in range(5)
        for start, end in ((60, 240), (360, 600))
    ]
    off_peak = []
    previous = 0
    for start, end in [*peak, [7 * 1440, 7 * 1440]]:
        if previous < start:
            off_peak.append([previous, start])
        previous = end
    result = []
    for model, (input_rate, cache_rate, output_rate) in OPENCODE_GO_SCHEDULED_MODELS.items():
        for period, windows, multiplier in (("peak", peak, 2), ("off_peak", off_peak, 1)):
            with localcontext() as context:
                context.prec = 64
                rates = {
                    "input_tokens": Decimal(input_rate) * multiplier,
                    "cache_read_tokens": Decimal(cache_rate) * multiplier,
                    "cache_write_tokens": Decimal(input_rate) * multiplier,
                    "output_tokens": Decimal(output_rate) * multiplier,
                    "reasoning_tokens": Decimal(output_rate) * multiplier,
                }
            result.append(
                UsagePrice(
                    price_id=f"current:go:{reviewed_at}:{model}:{period}",
                    provider="opencode_go",
                    model=model,
                    effective_from=0,
                    reviewed_at=reviewed_at,
                    valuation_basis="current_price_comparison",
                    source=OPENCODE_GO_SOURCE,
                    rates=rates,
                    constraints={
                        "modality": "text",
                        "service_tier": "standard",
                        "region": "global",
                        "modifiers": [],
                        "utc_weekly_intervals": [list(window) for window in windows],
                    },
                )
            )
    return tuple(result)


def matches_weekly_intervals(when: int | None, start: int | None, windows: object) -> bool:
    if when is None or not isinstance(windows, list):
        return False
    position = (when + 3 * 86_400_000) % (7 * 86_400_000)
    duration = when - start if start is not None else 0
    if windows[0][0] == 0 and windows[-1][1] == 10080:
        windows = [*windows, [windows[-1][0] - 10080, windows[0][1]]]
    return any(
        lower * 60_000 <= position - duration <= position < upper * 60_000
        for lower, upper in windows
    )


def validate_weekly_intervals(windows: object) -> None:
    if not isinstance(windows, list) or not windows or len(windows) > 32:
        raise ValueError("Invalid weekly price intervals")
    previous = 0
    for window in windows:
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("Invalid weekly price interval")
        start, end = window
        if type(start) is not int or type(end) is not int or not previous <= start < end <= 10080:
            raise ValueError("Invalid weekly price interval")
        previous = end
