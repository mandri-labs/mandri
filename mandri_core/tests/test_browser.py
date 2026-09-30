from unittest.mock import Mock

import pytest
from mandri.core import browser


@pytest.mark.parametrize(
    ("release", "expected"),
    [("6.6.87.2-microsoft-standard-WSL2", True), ("6.8.0-generic", False), ("10", False)],
)
def test_headless_posix_browser_opens_only_in_wsl(monkeypatch, release, expected):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser.platform, "release", lambda: release)
    monkeypatch.setattr(browser, "_available_openers", lambda: (("wslview",),))
    launch = Mock(return_value=True)
    monkeypatch.setattr(browser, "_run", launch)

    assert browser.open_browser("https://example.com") is expected
    if expected:
        launch.assert_called_once_with(("wslview", "https://example.com"))
    else:
        launch.assert_not_called()
