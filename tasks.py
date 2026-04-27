"""Transporter navigation: stations, motion primitives, docking, routes."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
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


# ── Default distances ───────────────────────────────────────────────
# Placeholder distances for the three rpi_build legs — small enough that
# the robot just nudges in each direction during bring-up. Replace with
# surveyed waypoints / SLAM goals once the route is mapped.
DEFAULT_DISTANCE_TO_ASSEMBLER_M = 0.05  # 5 cm forward
DEFAULT_DISTANCE_TO_DROP_OFF_M = 0.05  # 5 cm right
DEFAULT_DISTANCE_TO_BASE_M = 0.05  # 5 cm back (undo leg 1)


# ── Motion primitives ───────────────────────────────────────────────


def _drive(
    robot: "Transporter",
    *,
    x_vel: float,
    y_vel: float,
    distance_m: float,
    speed: float = 0.3,
) -> None:
    """Open-loop strafe in body frame.

    Drives at the unit-direction (x_vel, y_vel) scaled to `speed` for the
    time it takes to cover `distance_m`, then stops. No odometry —
    timing-based, fine for placeholder nudges, not for real navigation.
    """
    if distance_m <= 0:
        return
    norm = (x_vel * x_vel + y_vel * y_vel) ** 0.5
    if norm == 0:
        return
    ux, uy = x_vel / norm, y_vel / norm
    duration = distance_m / speed
    log.info(
        "Driving %.3f m at %.2f m/s along (%.2f, %.2f) (%.2f s)",
        distance_m, speed, ux, uy, duration,
    )
    robot.send_action({"x.vel": ux * speed, "y.vel": uy * speed, "theta.vel": 0.0})
    time.sleep(duration)
    robot.stop_base()
    log.info("Drive complete.")


def drive_forward(
    robot: "Transporter",
    distance_m: float,
    speed: float = 0.3,
) -> None:
    """Open-loop drive forward by *distance_m* metres at *speed* m/s."""
    _drive(robot, x_vel=1.0, y_vel=0.0, distance_m=distance_m, speed=speed)


# ── Docking (stub) ──────────────────────────────────────────────────


def dock_to_tag(
    robot: "Transporter", tag_id: int, timeout_s: float = 10.0
) -> None:
    """Visually servo onto an ArUco tag.

    STUB: pending camera + ArUco detector. Logs the target so the dispatch
    flow can be exercised end-to-end before perception is wired up.
    """
    log.info("dock_to_tag(tag_id=%d, timeout=%.1fs) — STUB", tag_id, timeout_s)


# ── rpi_build legs (placeholders) ──────────────────────────────────
# Each leg is a tiny nudge in a different direction so the operator
# can visually confirm the right command was received. They share
# `_drive` so swapping in real navigation later is a one-place change.


def go_to_assembler(
    robot: "Transporter",
    distance_m: float = DEFAULT_DISTANCE_TO_ASSEMBLER_M,
) -> None:
    """Leg 1/3 placeholder: nudge forward toward the assembler."""
    log.info("Leg 1/3: going to assembler (forward %.2f m placeholder)", distance_m)
    _drive(robot, x_vel=1.0, y_vel=0.0, distance_m=distance_m)


def go_to_drop_off(
    robot: "Transporter",
    distance_m: float = DEFAULT_DISTANCE_TO_DROP_OFF_M,
) -> None:
    """Leg 2/3 placeholder: nudge sideways (right) toward the drop-off.

    Body-frame y is left-positive, so right is `-y.vel`.
    """
    log.info("Leg 2/3: going to drop-off (right %.2f m placeholder)", distance_m)
    _drive(robot, x_vel=0.0, y_vel=-1.0, distance_m=distance_m)


def return_to_base(
    robot: "Transporter",
    distance_m: float = DEFAULT_DISTANCE_TO_BASE_M,
) -> None:
    """Leg 3/3 placeholder: nudge backward to home base."""
    log.info("Leg 3/3: returning to base (back %.2f m placeholder)", distance_m)
    _drive(robot, x_vel=-1.0, y_vel=0.0, distance_m=distance_m)


# ── Routes ──────────────────────────────────────────────────────────
# Each route is a plain function that drives the robot from one station to
# another. Stub bodies for now — geometry will be filled in once the routes
# have been measured on the floor.


RouteFn = Callable[["Transporter"], None]


def _picker_to_manipulator(robot: "Transporter") -> None:
    log.info("Route picker → manipulator (stub)")
    dock_to_tag(robot, STATION_TAG_IDS["manipulator"])


def _manipulator_to_delivery(robot: "Transporter") -> None:
    log.info("Route manipulator → delivery (stub)")
    dock_to_tag(robot, STATION_TAG_IDS["delivery"])


def _delivery_to_picker(robot: "Transporter") -> None:
    log.info("Route delivery → picker (stub)")
    dock_to_tag(robot, STATION_TAG_IDS["picker"])


ROUTES: dict[tuple[str, str], RouteFn] = {
    ("picker", "manipulator"): _picker_to_manipulator,
    ("manipulator", "delivery"): _manipulator_to_delivery,
    ("delivery", "picker"): _delivery_to_picker,
}


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
