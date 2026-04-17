"""Transporter task implementations — separated from communication logic."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transporter import Transporter

log = logging.getLogger(__name__)


def drive_forward(
    robot: "Transporter",
    distance_m: float,
    speed: float = 0.3,
) -> None:
    """Open-loop drive forward by *distance_m* metres at *speed* m/s.

    No odometry — just velocity for a calculated duration, then stop.
    """
    if distance_m <= 0:
        return
    duration = distance_m / speed
    log.info("Driving forward %.3f m at %.2f m/s (%.2f s)", distance_m, speed, duration)
    robot.send_action({"x.vel": speed, "y.vel": 0.0, "theta.vel": 0.0})
    time.sleep(duration)
    robot.stop_base()
    log.info("Drive complete.")
