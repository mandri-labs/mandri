"""Wall-clock helper shared across components."""

import time

from mandri.core.ids import EpochMs


def system_now_ms() -> EpochMs:
    return EpochMs(time.time_ns() // 1_000_000)
