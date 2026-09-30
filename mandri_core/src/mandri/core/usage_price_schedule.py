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
