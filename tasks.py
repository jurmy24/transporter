"""Transporter navigation: stations, motion primitives, docking, routes.

Configuration lives in JSON under config/, not in this file:

    config/markers.json   — every marker the robot knows about. Each entry
                            holds tag_id, size_m, and (optional) dock_pose.
                            teach-pose writes the dock_pose; tag_id/size_m
                            are set at registration time.
    config/routes.json    — multi-step routes keyed by "from -> to". Default
                            for any unlisted pair is one dock onto the
                            destination's station marker.

Both files are re-read on every run_route call so re-teaches and route edits
take effect without restarting factory_edge.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from perception import (
    ArucoDetector,
    Camera,
    CameraIntrinsics,
    CameraMount,
    DEFAULT_INTRINSICS_PATH,
    FrameStream,
    MjpegServer,
    camera_to_body,
)

# Optional shared MJPEG server. factory_edge sets this at startup if --stream-port
# was given; dock_to_tag pushes annotated frames here for browser preview.
mjpeg_server: MjpegServer | None = None

# Single shared camera reader. factory_edge owns the lifecycle; dock_to_tag
# pulls latest frames from it instead of opening its own VideoCapture.
frame_stream: FrameStream | None = None

# Set while dock_to_tag is in its control loop. The idle preview thread reads
# this to know whether to push its own annotated frames to MJPEG (when clear)
# or stay out of the way (when set, dock owns the stream output).
dock_active = threading.Event()

if TYPE_CHECKING:
    from transporter import Transporter

CONFIG_DIR = Path(__file__).parent / "config"
MARKERS_PATH = CONFIG_DIR / "markers.json"
ROUTES_PATH = CONFIG_DIR / "routes.json"

log = logging.getLogger(__name__)


# ── Stations ────────────────────────────────────────────────────────
# Stations are the subset of marker names the factory dispatches to/from.
# Marker definitions (tag_id, size, dock pose) live in markers.json.

STATIONS: tuple[str, ...] = ("picker", "manipulator", "delivery")


# ── Marker registry (markers.json) ──────────────────────────────────


@dataclass
class DockPose:
    """Desired robot pose relative to a marker, in body frame.

    forward_m  — body x distance from chassis centre to the marker.
    lateral_m  — body y offset (positive = marker should be to the robot's left).
    """

    forward_m: float
    lateral_m: float = 0.0


@dataclass
class Marker:
    name: str
    tag_id: int
    size_m: float
    dock_pose: DockPose | None = None


def load_markers() -> dict[str, Marker]:
    """Read every registered marker from config/markers.json."""
    if not MARKERS_PATH.exists():
        return {}
    data = json.loads(MARKERS_PATH.read_text())
    out: dict[str, Marker] = {}
    for name, info in data.items():
        pose_data = info.get("dock_pose")
        pose = DockPose(**pose_data) if pose_data else None
        out[name] = Marker(
            name=name,
            tag_id=int(info["tag_id"]),
            size_m=float(info["size_m"]),
            dock_pose=pose,
        )
    return out


def _save_markers(markers: dict[str, Marker]) -> None:
    serialised = {
        name: {
            "tag_id": m.tag_id,
            "size_m": m.size_m,
            "dock_pose": asdict(m.dock_pose) if m.dock_pose else None,
        }
        for name, m in markers.items()
    }
    MARKERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKERS_PATH.write_text(json.dumps(serialised, indent=2) + "\n")


def register_marker(name: str, tag_id: int, size_m: float) -> Marker:
    """Add or update a marker definition. Preserves dock_pose if present."""
    markers = load_markers()
    existing_pose = markers[name].dock_pose if name in markers else None
    markers[name] = Marker(
        name=name, tag_id=tag_id, size_m=size_m, dock_pose=existing_pose
    )
    _save_markers(markers)
    return markers[name]


def save_marker_pose(name: str, pose: DockPose) -> None:
    """Update the dock_pose of an already-registered marker."""
    markers = load_markers()
    if name not in markers:
        raise KeyError(
            f"marker '{name}' not registered. Pass --tag-id and --size-m on first teach."
        )
    markers[name].dock_pose = pose
    _save_markers(markers)


# Built-in compatibility export for factory_edge.py: the tag IDs of the
# stations, taken from markers.json at startup.
def _station_tag_ids() -> dict[str, int]:
    markers = load_markers()
    return {name: markers[name].tag_id for name in STATIONS if name in markers}


STATION_TAG_IDS: dict[str, int] = _station_tag_ids()


# ── Default distances (rpi_build legs) ─────────────────────────────
DEFAULT_DISTANCE_TO_ASSEMBLER_M = 0.05
DEFAULT_DISTANCE_TO_DROP_OFF_M = 0.05
DEFAULT_DISTANCE_TO_BASE_M = 0.05


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
    time it takes to cover `distance_m`, then stops. Timing-based, fine
    for placeholder nudges, not for real navigation.
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
    _send_velocity(robot, ux * speed, uy * speed, 0.0)
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


_ROTATE_ANG_VEL = 0.4  # rad/s — open-loop in-place rotation speed


def _rotate(
    robot: "Transporter",
    yaw_rad: float,
    ang_vel: float = _ROTATE_ANG_VEL,
) -> None:
    """Open-loop in-place rotation by `yaw_rad`.

    Convention: positive yaw = CCW (left turn from the robot's perspective),
    negative = CW (right turn). Timing-based; few-degree error expected, the
    next dock step refines.
    """
    if yaw_rad == 0.0:
        return
    duration = abs(yaw_rad) / ang_vel
    sign = 1.0 if yaw_rad > 0 else -1.0
    log.info(
        "Rotating %.1f° at %.2f rad/s (%.2f s)",
        math.degrees(yaw_rad), ang_vel, duration,
    )
    _send_velocity(robot, 0.0, 0.0, sign * ang_vel)
    time.sleep(duration)
    robot.stop_base()
    log.info("Rotate complete.")


# ── Docking ─────────────────────────────────────────────────────────


# Tunables for the closed-loop dock controller.
#
# Control strategy: continuous P-control on body-frame error at LOOP_HZ.
# Translation and yaw run in two phases — translate first, then yaw — because
# during translation the bearing to the marker swings as the robot moves
# laterally, making yaw error noisy and prone to driving spin-strafe coupling.
#
# Convergence rules to avoid orbiting:
#   1. Hard deadband: inside POSITION_TOL_M, command exactly zero (don't try
#      to drive sub-cm corrections — motor coast is bigger than the command).
#   2. Tolerance ≥ coast distance: at MAX_LIN_VEL, the chassis coasts ~2 cm
#      after a stop command, so POSITION_TOL_M must be ≥ that. Tighter than
#      this is what caused the previous orbit/limit-cycle.
#   3. Settle-hold: the in-tol condition must persist ARRIVED_HOLD_S before
#      declaring arrived, so coast that drifts back out resumes control.
#   4. Stiction floor: the wheels don't turn reliably below ~0.04 m/s, so any
#      nonzero command is bumped up to MIN_LIN_VEL.
_DOCK_LOOP_HZ = 20.0
_DOCK_GRACE_S = 2.0
_DOCK_LOST_TIMEOUT_S = 10.0
_DOCK_ARRIVED_HOLD_S = 0.5
_DOCK_POSITION_TOL_M = 0.02    # 2 cm — must exceed natural coast at MAX_LIN_VEL
# Heading tolerance must exceed the rvec-ambiguity noise band of solvePnP for
# near-fronto-parallel views. Empirically that band is ±~10° on this rig: the
# normal vector flips between two valid solutions on small corner-detection
# noise. Tighter than ~6° caused chronic REFINE/ALIGN flapping at the goal,
# wasting seconds with the robot already physically squared up.
_DOCK_YAW_TOL_RAD = math.radians(6.0)
_DOCK_KP_LIN = 0.8             # m/s per m of position error
_DOCK_KP_YAW = 1.0             # rad/s per rad of heading error
_DOCK_MAX_LIN_VEL = 0.12       # m/s — cap on translation speed
_DOCK_MIN_LIN_VEL = 0.04       # m/s — wheels stall below this, snap up to it
_DOCK_MAX_ANG_VEL = 0.4        # rad/s
_DOCK_MIN_ANG_VEL = 0.2        # rad/s — yaw stiction floor
_DOCK_SEARCH_ANG_VEL = 0.3     # rad/s while rotating to find a lost marker
# Above this heading error we square up the chassis BEFORE translating: a
# rotation in place at 0.5 m range moves the marker ~r·θ in body frame
# (chassis pivots, marker doesn't), so translating with a big heading error
# means chasing a target that the next rotate phase will yank away. Below
# this threshold the disturbance is small enough that it's cheaper to just
# translate first and clean up heading at the end. Set above the rvec
# ambiguity noise band (~±10°) so a noisy single frame doesn't trip a full
# ALIGN rotation when the chassis is already nearly squared up.
_DOCK_LARGE_HEADING_RAD = math.radians(15.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _send_velocity(robot: "Transporter", vx: float, vy: float, wz_rad_s: float) -> None:
    """Send a body-frame velocity to the Transporter, in SI units.

    Transporter.send_action expects theta.vel in DEGREES/s (see
    transporter.py:_body_to_wheel_raw which converts deg→rad internally).
    All our control code thinks in rad/s, so we convert here in one place.
    """
    robot.send_action(
        {"x.vel": vx, "y.vel": vy, "theta.vel": math.degrees(wz_rad_s)}
    )


def _push_stream(
    frame: np.ndarray,
    detector: "ArucoDetector",
    detections: list,
    tag_id: int,
    *,
    phase: str,
    forward_err: float | None,
    lateral_err: float | None,
    heading_err: float | None,
    intrinsics: "CameraIntrinsics",
    mount: "CameraMount",
) -> None:
    """Annotate a frame and push it to the optional MJPEG server.

    Cheap when the server is unset (just an early return). When set, draws
    detection outlines + axes + a HUD with phase, errors, and the target
    tag id, then JPEG-encodes and hands it off (encoding ~5 ms at 720p).
    """
    if mjpeg_server is None:
        return
    overlay = frame.copy()
    detector.draw(overlay, detections)
    target_seen = any(d.id == tag_id for d in detections)
    hud_lines = [f"phase={phase}  target=tag{tag_id} {'OK' if target_seen else 'LOST'}"]
    if forward_err is not None and lateral_err is not None:
        hud_lines.append(f"fwd={forward_err:+.3f}m  lat={lateral_err:+.3f}m")
    if heading_err is not None:
        hud_lines.append(f"hdg={math.degrees(heading_err):+.2f}deg")
    colour = (0, 255, 0) if target_seen else (0, 0, 255)
    for i, line in enumerate(hud_lines):
        cv2.putText(
            overlay, line, (10, 30 + 26 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
        )
    if mount.rotated_180:
        overlay = cv2.rotate(overlay, cv2.ROTATE_180)
    mjpeg_server.push(overlay)


def dock_to_tag(
    robot: "Transporter",
    tag_id: int,
    marker_size_m: float,
    dock_pose: DockPose,
    timeout_s: float = 30.0,
    search_velocity: tuple[float, float, float] | None = None,
) -> None:
    """Visually servo to `dock_pose` relative to marker `tag_id`.

    `marker_size_m` is the printed edge length; pose accuracy depends on it
    being correct, so different-sized markers in the same route each get
    their actual size.

    `search_velocity`, if given, is a body-frame (vx, vy, wz) used DURING the
    initial SEEK only — i.e., before the marker has been detected even once.
    Useful when the marker isn't initially in view and the robot must drive
    in a known direction to acquire it (e.g., strafe right). After the first
    detection, search reverts to the default rotate-to-search behaviour so a
    transient detection dropout near the goal doesn't push the robot off.

    State machine:
        SEEK       — marker not yet in view. Wait passively for _DOCK_GRACE_S,
                     then either run search_velocity (if given and never yet
                     seen) or rotate slowly. Raise after _DOCK_LOST_TIMEOUT_S.
        ALIGN      — heading error > _DOCK_LARGE_HEADING_RAD: rotate in place
                     to square up before any translation.
        TRANSLATE  — heading roughly square: P-control on body-frame position.
        REFINE     — in position, finish the heading correction.
        ARRIVED    — all three within tolerance for _DOCK_ARRIVED_HOLD_S.

    Heading is measured from the marker's rvec (its plane normal in body
    frame), not from where it sits in the camera view — so a robot that's
    in the right (x, y) but rotated still gets squared up.
    """
    intrinsics = CameraIntrinsics.load(DEFAULT_INTRINSICS_PATH)
    mount = CameraMount.load()
    detector = ArucoDetector(marker_size_m=marker_size_m, intrinsics=intrinsics)
    R_body_cam = mount.rotation_body_from_camera()  # cached: depends only on mount

    # Loop rate is now driven by the FrameStream — we run as fast as the
    # camera produces frames (~30 Hz). _DOCK_LOOP_HZ is kept only as a
    # historical reference; nothing reads it inside the loop anymore.
    start_t = time.monotonic()
    last_seen_t = start_t
    last_diag_t = start_t
    arrived_since: float | None = None
    base_idle = False  # tracks whether the wheels are already commanded to zero
    marker_ever_seen = False  # gates the use of search_velocity
    DIAG_INTERVAL_S = 1.0

    def _idle_once() -> None:
        nonlocal base_idle
        if not base_idle:
            robot.stop_base()
            base_idle = True

    def _command(vx: float, vy: float, wz: float) -> None:
        nonlocal base_idle
        _send_velocity(robot, vx, vy, wz)
        base_idle = vx == 0.0 and vy == 0.0 and wz == 0.0

    log.info(
        "dock_to_tag(tag=%d, marker=%.0fmm, dock=(fwd=%.3fm, lat=%.3fm), timeout=%.1fs)",
        tag_id, marker_size_m * 1000, dock_pose.forward_m, dock_pose.lateral_m, timeout_s,
    )

    if frame_stream is None:
        raise RuntimeError(
            "dock_to_tag requires tasks.frame_stream to be set; "
            "factory_edge should initialize it before dispatch."
        )

    dock_active.set()
    last_frame_t = 0.0
    try:
        while True:
            now = time.monotonic()

            if now - start_t > timeout_s:
                _idle_once()
                raise TimeoutError(
                    f"dock_to_tag(tag={tag_id}) timed out after {timeout_s:.1f}s"
                )

            # Block until a fresher frame than we last processed; the
            # producer thread runs at camera FPS (~30 Hz), so this is also
            # what paces our control loop.
            res = frame_stream.wait_next(after_t=last_frame_t, timeout=1.0)
            if res is None:
                # Producer is wedged or paused — treat like a missed frame.
                gap = now - last_seen_t
                if gap >= _DOCK_LOST_TIMEOUT_S:
                    _idle_once()
                    raise RuntimeError(
                        f"dock_to_tag(tag={tag_id}): no frames for {gap:.1f}s"
                    )
                _idle_once()
                continue
            frame, last_frame_t = res
            now = time.monotonic()

            detections = detector.detect(frame)
            target = next((d for d in detections if d.id == tag_id), None)

            if target is None or target.tvec is None or target.rvec is None:
                gap = now - last_seen_t
                if gap >= _DOCK_LOST_TIMEOUT_S:
                    _idle_once()
                    raise RuntimeError(
                        f"dock_to_tag(tag={tag_id}): marker not seen for {gap:.1f}s"
                    )
                if gap < _DOCK_GRACE_S:
                    _idle_once()  # passive look — silent after the first stop
                elif search_velocity is not None and not marker_ever_seen:
                    # User-supplied initial search direction (e.g., strafe
                    # right to bring an off-frame marker into view).
                    _command(*search_velocity)
                else:
                    _command(0.0, 0.0, _DOCK_SEARCH_ANG_VEL)
                arrived_since = None
                _push_stream(
                    frame, detector, detections, tag_id,
                    phase="SEEK", forward_err=None, lateral_err=None,
                    heading_err=None, intrinsics=intrinsics, mount=mount,
                )
                continue

            last_seen_t = now
            marker_ever_seen = True
            x_b, y_b, _z = camera_to_body(target.tvec, mount)

            forward_err = x_b - dock_pose.forward_m
            lateral_err = y_b - dock_pose.lateral_m

            # Heading error: marker outward normal in body frame.
            #     n_body = R_body_cam · R_cam_marker · (0, 0, +1)
            # solvePnP for a face-on marker returns rvec ≈ (π, 0, 0): the
            # marker has +y up in its own frame but the camera has +y down,
            # so face-on already encodes a 180° flip about cam-x. With that
            # baked into rvec, the marker's local +z is the outward normal
            # (toward the camera). When the chassis is squared up,
            # n_body = (−1, 0, 0). For a chassis rotated CCW by α the
            # outward normal in body becomes (−cos α, sin α, 0), so
            # atan2(−n_y, −n_x) returns −α — wz = KP·err is negative → CW
            # correction (the right way to undo a CCW over-rotation).
            # Verified from first-principles transforms, not from rvec
            # composition: an earlier sign mistake in that compositional
            # smoke test cost two iterations.
            R_cam_marker, _ = cv2.Rodrigues(target.rvec.reshape(3, 1))
            n_body = R_body_cam @ R_cam_marker @ np.array([0.0, 0.0, 1.0])
            heading_err = math.atan2(-n_body[1], -n_body[0])

            in_position = (
                abs(forward_err) < _DOCK_POSITION_TOL_M
                and abs(lateral_err) < _DOCK_POSITION_TOL_M
            )
            in_heading = abs(heading_err) < _DOCK_YAW_TOL_RAD
            heading_is_large = abs(heading_err) > _DOCK_LARGE_HEADING_RAD

            # Phase priority — see the docstring. Big heading first (else
            # translation chases a target the next rotate phase would yank
            # away), then translate, then heading refine.
            if in_position and in_heading:
                phase = "HOLD"
            elif heading_is_large:
                phase = "ALIGN"
            elif not in_position:
                phase = "TRANSLATE"
            else:
                phase = "REFINE"

            if now - last_diag_t >= DIAG_INTERVAL_S:
                log.info(
                    "[dock] phase=%s fwd=%+.4fm lat=%+.4fm hdg=%+.2f° (x_b=%.3f y_b=%.3f)",
                    phase, forward_err, lateral_err, math.degrees(heading_err), x_b, y_b,
                )
                last_diag_t = now

            _push_stream(
                frame, detector, detections, tag_id,
                phase=phase,
                forward_err=forward_err, lateral_err=lateral_err,
                heading_err=heading_err, intrinsics=intrinsics, mount=mount,
            )

            if phase == "HOLD":
                if arrived_since is None:
                    arrived_since = now
                if now - arrived_since >= _DOCK_ARRIVED_HOLD_S:
                    _idle_once()
                    log.info(
                        "Arrived at tag=%d  fwd_err=%.4fm lat_err=%.4fm hdg_err=%.2f°",
                        tag_id, forward_err, lateral_err, math.degrees(heading_err),
                    )
                    return
                # Hold idle while we wait for the in-tol band to persist.
                _idle_once()
                continue

            arrived_since = None
            if phase in ("ALIGN", "REFINE"):
                # Pure rotation. Continuous P-control with stiction floor.
                wz_raw = _DOCK_KP_YAW * heading_err
                wz_abs = abs(wz_raw)
                if wz_abs > _DOCK_MAX_ANG_VEL:
                    wz = math.copysign(_DOCK_MAX_ANG_VEL, wz_raw)
                elif wz_abs < _DOCK_MIN_ANG_VEL:
                    wz = math.copysign(_DOCK_MIN_ANG_VEL, wz_raw)
                else:
                    wz = wz_raw
                _command(0.0, 0.0, wz)
            else:
                # Pure translation. P-control on (forward_err, lateral_err)
                # with magnitude clamped to MAX and bumped to MIN if nonzero.
                vx_raw = _DOCK_KP_LIN * forward_err
                vy_raw = _DOCK_KP_LIN * lateral_err
                v_mag = math.hypot(vx_raw, vy_raw)
                if v_mag > _DOCK_MAX_LIN_VEL:
                    scale = _DOCK_MAX_LIN_VEL / v_mag
                elif v_mag < _DOCK_MIN_LIN_VEL:
                    scale = _DOCK_MIN_LIN_VEL / v_mag
                else:
                    scale = 1.0
                _command(vx_raw * scale, vy_raw * scale, 0.0)
    finally:
        dock_active.clear()


# ── rpi_build legs (placeholders) ──────────────────────────────────


def go_to_assembler(
    robot: "Transporter",
    distance_m: float = DEFAULT_DISTANCE_TO_ASSEMBLER_M,
) -> None:
    log.info("Leg 1/3: going to assembler (forward %.2f m placeholder)", distance_m)
    _drive(robot, x_vel=1.0, y_vel=0.0, distance_m=distance_m)


def go_to_drop_off(
    robot: "Transporter",
    distance_m: float = DEFAULT_DISTANCE_TO_DROP_OFF_M,
) -> None:
    log.info("Leg 2/3: going to drop-off (right %.2f m placeholder)", distance_m)
    _drive(robot, x_vel=0.0, y_vel=-1.0, distance_m=distance_m)


def return_to_base(
    robot: "Transporter",
    distance_m: float = DEFAULT_DISTANCE_TO_BASE_M,
) -> None:
    log.info("Leg 3/3: returning to base (back %.2f m placeholder)", distance_m)
    _drive(robot, x_vel=-1.0, y_vel=0.0, distance_m=distance_m)


# ── Routes ──────────────────────────────────────────────────────────
# A route is a sequence of typed steps. Steps are loaded from
# config/routes.json; new step types only need a parser and an executor.


@dataclass(frozen=True)
class DockStep:
    """Visual-servo dock onto a registered marker. Pose comes from markers.json.

    `search_vx/vy/wz` (body-frame m/s, rad/s) override the SEEK behaviour for
    cases where the marker isn't initially in view — e.g., the manipulator→picker
    leg starts with the marker out of frame and must strafe right to acquire.
    Default zero ⇒ fall back to the rotate-search.
    """
    marker_name: str
    search_vx: float = 0.0
    search_vy: float = 0.0
    search_wz: float = 0.0


@dataclass(frozen=True)
class RotateStep:
    """Open-loop rotation. yaw_rad > 0 = left (CCW), < 0 = right (CW)."""
    yaw_rad: float


@dataclass(frozen=True)
class DriveStep:
    """Open-loop body-frame translation. (x_m, y_m) is the displacement vector."""
    x_m: float
    y_m: float
    speed: float = 0.3


Step = DockStep | RotateStep | DriveStep


def _parse_step(d: dict[str, Any]) -> Step:
    t = d.get("type")
    if t == "dock":
        return DockStep(
            marker_name=d["marker"],
            search_vx=float(d.get("search_vx", 0.0)),
            search_vy=float(d.get("search_vy", 0.0)),
            search_wz=float(d.get("search_wz", 0.0)),
        )
    if t == "rotate":
        return RotateStep(yaw_rad=math.radians(float(d["yaw_deg"])))
    if t == "drive":
        return DriveStep(
            x_m=float(d.get("x_m", 0.0)),
            y_m=float(d.get("y_m", 0.0)),
            speed=float(d.get("speed", 0.3)),
        )
    raise ValueError(f"unknown step type {t!r} in routes.json")


def load_routes() -> dict[tuple[str, str], list[Step]]:
    """Read all routes from config/routes.json. Keys are 'from -> to' strings."""
    if not ROUTES_PATH.exists():
        return {}
    data = json.loads(ROUTES_PATH.read_text())
    routes: dict[tuple[str, str], list[Step]] = {}
    for key, raw_steps in data.items():
        if " -> " not in key:
            log.warning("Skipping route key %r (expected 'from -> to')", key)
            continue
        frm, to = (s.strip() for s in key.split(" -> ", 1))
        routes[(frm, to)] = [_parse_step(s) for s in raw_steps]
    return routes


def _validate(steps: list[Step], markers: dict[str, Marker]) -> None:
    """Pre-flight: ensure every DockStep references a registered marker with a pose."""
    for step in steps:
        if isinstance(step, DockStep):
            if step.marker_name not in markers:
                raise ValueError(
                    f"route references unknown marker {step.marker_name!r}. "
                    f"Add it to config/markers.json."
                )
            if markers[step.marker_name].dock_pose is None:
                raise ValueError(
                    f"marker {step.marker_name!r} has no taught pose. Run: "
                    f"python -m perception teach-pose --marker {step.marker_name}"
                )


def run_route(robot: "Transporter", from_station: str, to_station: str) -> None:
    """Execute one delivery leg as a sequence of route steps."""
    if from_station not in STATIONS:
        raise ValueError(f"unknown from_station: {from_station}")
    if to_station not in STATIONS:
        raise ValueError(f"unknown to_station: {to_station}")

    markers = load_markers()
    routes = load_routes()

    steps = routes.get((from_station, to_station))
    if steps is None:
        steps = [DockStep(to_station)]  # default: single dock onto destination

    _validate(steps, markers)

    log.info("Route %s → %s: %d step(s)", from_station, to_station, len(steps))
    for i, step in enumerate(steps, 1):
        if isinstance(step, DockStep):
            m = markers[step.marker_name]
            sv = (step.search_vx, step.search_vy, step.search_wz)
            search = sv if any(sv) else None
            log.info(
                "  step %d/%d: dock to '%s' (tag=%d, %.0fmm)%s",
                i, len(steps), step.marker_name, m.tag_id, m.size_m * 1000,
                f"  search_vel=({sv[0]:+.2f}, {sv[1]:+.2f}, {sv[2]:+.2f})" if search else "",
            )
            assert m.dock_pose is not None  # validated above
            dock_to_tag(robot, m.tag_id, m.size_m, m.dock_pose, search_velocity=search)
        elif isinstance(step, RotateStep):
            log.info("  step %d/%d: rotate %.1f°", i, len(steps), math.degrees(step.yaw_rad))
            _rotate(robot, step.yaw_rad)
        elif isinstance(step, DriveStep):
            log.info(
                "  step %d/%d: drive (%.2f, %.2f) m at %.2f m/s",
                i, len(steps), step.x_m, step.y_m, step.speed,
            )
            distance = (step.x_m * step.x_m + step.y_m * step.y_m) ** 0.5
            _drive(robot, x_vel=step.x_m, y_vel=step.y_m,
                   distance_m=distance, speed=step.speed)
    log.info("Arrived at %s", to_station)
