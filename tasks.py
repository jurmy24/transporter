"""Transporter navigation: stations, motion primitives, docking, routes."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transporter import Transporter

log = logging.getLogger(__name__)


# ── Stations ────────────────────────────────────────────────────────
# Single source of truth for ArUco tag IDs. Drawn from DICT_4X4_50;
# IDs are kept sparse so a misread is unlikely to resolve to another station.

STATION_TAG_IDS: dict[str, int] = {
    "picker": 10,
    "manipulator": 20,
    "delivery": 30,
}


# ── Motion primitives ───────────────────────────────────────────────


def drive_forward(
    robot: "Transporter", distance_m: float, speed: float = 0.3
) -> None:
    """Open-loop forward drive for a calculated duration."""
    if distance_m <= 0:
        return
    duration = distance_m / speed
    log.info("Forward %.3f m @ %.2f m/s (%.2f s)", distance_m, speed, duration)
    robot.send_action({"x.vel": speed, "y.vel": 0.0, "theta.vel": 0.0})
    time.sleep(duration)
    robot.stop_base()


# ── Docking (stub) ──────────────────────────────────────────────────


def dock_to_tag(
    robot: "Transporter", tag_id: int, timeout_s: float = 10.0
) -> None:
    """Visually servo onto an ArUco tag.

    STUB: pending camera + ArUco detector. Logs the target so the dispatch
    flow can be exercised end-to-end before perception is wired up.
    """
    log.info("dock_to_tag(tag_id=%d, timeout=%.1fs) — STUB", tag_id, timeout_s)


# ── Routes ──────────────────────────────────────────────────────────
# Each route is a plain function that drives the robot from one station to
# another. Stub bodies for now — geometry will be filled in once the routes
# have been measured on the floor.


def _picker_to_manipulator(robot: "Transporter") -> None:
    log.info("Route picker → manipulator (stub)")
    dock_to_tag(robot, STATION_TAG_IDS["manipulator"])


def _manipulator_to_delivery(robot: "Transporter") -> None:
    log.info("Route manipulator → delivery (stub)")
    dock_to_tag(robot, STATION_TAG_IDS["delivery"])


def _delivery_to_picker(robot: "Transporter") -> None:
    log.info("Route delivery → picker (stub)")
    dock_to_tag(robot, STATION_TAG_IDS["picker"])


ROUTES: dict[tuple[str, str], "RouteFn"] = {
    ("picker", "manipulator"): _picker_to_manipulator,
    ("manipulator", "delivery"): _manipulator_to_delivery,
    ("delivery", "picker"): _delivery_to_picker,
}


from collections.abc import Callable  # noqa: E402

RouteFn = Callable[["Transporter"], None]


def run_route(robot: "Transporter", from_station: str, to_station: str) -> None:
    if from_station not in STATION_TAG_IDS:
        raise ValueError(f"unknown from_station: {from_station}")
    if to_station not in STATION_TAG_IDS:
        raise ValueError(f"unknown to_station: {to_station}")
    route = ROUTES.get((from_station, to_station))
    if route is None:
        raise ValueError(f"no route defined: {from_station} → {to_station}")
    log.info("Running route %s → %s", from_station, to_station)
    route(robot)
    log.info("Arrived at %s", to_station)
