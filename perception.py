"""Camera + ArUco perception, plus dev CLI.

Library surface:
    Camera                  — context-manager wrapper around cv2.VideoCapture.
    CameraIntrinsics        — load/save 3x3 K and distortion coefficients.
    CameraMount             — static rigid transform body↔camera (mount geom).
    camera_to_body          — map a camera-frame point (3,) to body frame.
    ArucoDetector           — DICT_4X4_50 detection; pose if intrinsics given.
    MarkerDetection         — id, image-frame corners, optional rvec/tvec.

Frame conventions:
    Body            x = forward, y = left, z = up (right-handed).
    Camera optical  x = right,   y = down, z = forward (OpenCV / out of lens).

CLI (run from the repo root):
    python -m perception generate            # PNGs of station markers
    python -m perception make-board          # printable ChArUco for US Letter
    python -m perception calibrate           # headless ChArUco calibration
    python -m perception detect              # live detection (browser preview)

Hardware-specific config lives in config/:
    camera_intrinsics.json   — focal length, principal point, distortion
    camera_mount.json        — forward offset, pitch, upside-down flag
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

ARUCO_DICT = cv2.aruco.DICT_4X4_50

CONFIG_DIR = Path(__file__).parent / "config"
DEFAULT_INTRINSICS_PATH = CONFIG_DIR / "camera_intrinsics.json"
DEFAULT_MOUNT_PATH = CONFIG_DIR / "camera_mount.json"


# ── Library ─────────────────────────────────────────────────────────


@dataclass
class CameraIntrinsics:
    matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]

    @classmethod
    def load(cls, path: Path = DEFAULT_INTRINSICS_PATH) -> "CameraIntrinsics":
        data = json.loads(path.read_text())
        return cls(
            matrix=np.array(data["matrix"], dtype=np.float64),
            distortion=np.array(data["distortion"], dtype=np.float64),
            image_size=tuple(data["image_size"]),
        )

    def save(self, path: Path = DEFAULT_INTRINSICS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "matrix": self.matrix.tolist(),
                    "distortion": self.distortion.tolist(),
                    "image_size": list(self.image_size),
                },
                indent=2,
            )
        )


@dataclass
class CameraMount:
    """Static rigid transform from camera optical frame to body frame.

    forward_offset_m  — distance from chassis centre to camera origin, along body +x.
    pitch_deg         — camera pitched up by this many degrees (>0 = lens tilts up).
    rotated_180       — camera physically rotated 180° around its optical Z axis
                        (i.e., the image is upside-down). Calibration intrinsics
                        stay valid; this flag accounts for the orientation when
                        mapping marker poses into body frame.
    """

    forward_offset_m: float = 0.0
    pitch_deg: float = 0.0
    rotated_180: bool = False

    @classmethod
    def load(cls, path: Path = DEFAULT_MOUNT_PATH) -> "CameraMount":
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))

    def save(self, path: Path = DEFAULT_MOUNT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "forward_offset_m": self.forward_offset_m,
                    "pitch_deg": self.pitch_deg,
                    "rotated_180": self.rotated_180,
                },
                indent=2,
            )
        )

    def rotation_body_from_camera(self) -> np.ndarray:
        """3x3 rotation R such that p_body = R @ p_camera_optical."""
        # Right-side-up, no-tilt baseline (camera looks along body +x):
        #   cam +X (image right) → body -Y (right of robot)
        #   cam +Y (image down)  → body -Z (down)
        #   cam +Z (forward)     → body +X (forward)
        R = np.array(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            dtype=float,
        )
        if self.rotated_180:
            # 180° about optical Z: image-right and image-down both flip
            # relative to the world. Compose in the camera frame (right-multiply).
            R = R @ np.diag([-1.0, -1.0, 1.0])
        # Pitch around body +Y axis. Positive pitch_deg = camera tilted upward,
        # so body-frame +x moves toward +z.
        a = math.radians(self.pitch_deg)
        c, s = math.cos(a), math.sin(a)
        R_pitch = np.array(
            [[c, 0, -s], [0, 1, 0], [s, 0, c]],
            dtype=float,
        )
        return R_pitch @ R


def camera_to_body(tvec_cam: np.ndarray, mount: CameraMount) -> np.ndarray:
    """Map a camera-frame point (3,) to body frame (3,).

    Returns (x_forward, y_left, z_up) in metres relative to chassis centre.
    """
    R = mount.rotation_body_from_camera()
    t_offset = np.array([mount.forward_offset_m, 0.0, 0.0])
    return R @ np.asarray(tvec_cam, dtype=float) + t_offset


class Camera:
    """VideoCapture wrapper. Defaults match the Innomaker U20CAM on /dev/video0."""

    def __init__(
        self,
        device: str | int = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("Camera not open")
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_a) -> None:
        self.close()


@dataclass
class MarkerDetection:
    id: int
    corners: np.ndarray  # (4, 2) image-frame pixels, TL-TR-BR-BL
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None

    @property
    def distance_m(self) -> float | None:
        return float(np.linalg.norm(self.tvec)) if self.tvec is not None else None

    @property
    def centre_px(self) -> np.ndarray:
        return self.corners.mean(axis=0)


class ArucoDetector:
    def __init__(
        self,
        marker_size_m: float = 0.036,
        intrinsics: CameraIntrinsics | None = None,
    ) -> None:
        self.marker_size_m = marker_size_m
        self.intrinsics = intrinsics
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        s = marker_size_m / 2.0
        # cv2.aruco corner order: TL, TR, BR, BL on the marker plane (z=0).
        self._object_points = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
            dtype=np.float32,
        )

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        results: list[MarkerDetection] = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            pts = marker_corners.reshape(4, 2)
            det = MarkerDetection(id=int(marker_id), corners=pts)
            if self.intrinsics is not None:
                ok, rvec, tvec = cv2.solvePnP(
                    self._object_points,
                    pts.astype(np.float32),
                    self.intrinsics.matrix,
                    self.intrinsics.distortion,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok:
                    det.rvec = rvec.flatten()
                    det.tvec = tvec.flatten()
            results.append(det)
        return results

    def draw(self, frame: np.ndarray, detections: list[MarkerDetection]) -> np.ndarray:
        for d in detections:
            pts = d.corners.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            label = f"id={d.id}"
            if d.distance_m is not None:
                label += f" {d.distance_m:.2f}m"
            cv2.putText(
                frame,
                label,
                tuple(d.centre_px.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            if d.rvec is not None and self.intrinsics is not None:
                cv2.drawFrameAxes(
                    frame,
                    self.intrinsics.matrix,
                    self.intrinsics.distortion,
                    d.rvec,
                    d.tvec,
                    self.marker_size_m * 0.5,
                )
        return frame


# ── MJPEG streaming (for headless preview over SSH) ─────────────────


class MjpegServer:
    """Minimal MJPEG-over-HTTP server. Push annotated frames; view in a browser.

    Single endpoint at `/` serves multipart/x-mixed-replace, which every
    modern browser renders as a continuously-updating image. Only the most
    recent frame is held — late readers don't replay history.
    """

    def __init__(self, port: int, jpeg_quality: int = 70) -> None:
        self.port = port
        self._jpeg_quality = jpeg_quality
        self._latest: bytes | None = None
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def push(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if ok:
            with self._lock:
                self._latest = buf.tobytes()

    def _get_latest(self) -> bytes | None:
        with self._lock:
            return self._latest

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_kw):
                return  # silence per-request stderr noise

            def do_GET(self) -> None:
                if self.path not in ("/", "/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                try:
                    while True:
                        data = outer._get_latest()
                        if data:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(data)}\r\n\r\n".encode()
                            )
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        log.info("MJPEG stream live on http://<this-host>:%d/", self.port)


# ── Shared camera frame producer ───────────────────────────────────


class FrameStream:
    """One-thread owner of a Camera. Multiple consumers pull the latest frame.

    cv2.VideoCapture is not safe for concurrent readers, and Linux v4l2
    typically only allows one open at a time. So at runtime there's exactly
    one of these on the Pi: it owns /dev/video0, runs a tight read loop in
    a background thread, and notifies waiters via a condition variable.

    Consumers (the dock controller, the idle preview) call wait_next() with
    the timestamp of the last frame they processed; they block until a newer
    frame arrives, then run their own detection / annotation on it.
    """

    def __init__(self, camera: "Camera") -> None:
        self._cam = camera
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._cam.open()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="FrameStream"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cam.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._cam.read()
            except Exception as e:
                log.warning("FrameStream camera read failed: %s", e)
                time.sleep(0.05)
                continue
            with self._cond:
                self._frame = frame
                self._t = time.monotonic()
                self._cond.notify_all()

    def wait_next(
        self, after_t: float = 0.0, timeout: float = 1.0
    ) -> tuple[np.ndarray, float] | None:
        """Block until a frame newer than `after_t` is available, or timeout.

        Returns (frame_copy, capture_timestamp) or None on timeout.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None or self._t <= after_t:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return None
                self._cond.wait(timeout=rem)
            return self._frame.copy(), self._t


# ── CLI subcommands ─────────────────────────────────────────────────


def _cmd_generate(args: argparse.Namespace) -> None:
    """Write printable PNGs for the station markers."""
    from tasks import STATION_TAG_IDS

    args.out.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    for name, tag_id in STATION_TAG_IDS.items():
        img = cv2.aruco.generateImageMarker(dictionary, tag_id, args.pixels)
        path = args.out / f"aruco_{tag_id:02d}_{name}.png"
        cv2.imwrite(str(path), img)
        print(f"Wrote {path}")


def _cmd_make_board(args: argparse.Namespace) -> None:
    """Generate a print-ready ChArUco calibration target for US Letter portrait.

    A 100 mm scale bar is rendered below the board: measure it with a
    ruler after printing to confirm your printer didn't auto-scale.
    """
    DPI = 300
    PAGE_MM = (215.9, 279.4)  # US Letter portrait
    MARGIN_MM = 8.0

    cols, rows = args.cols, args.rows
    square_mm = args.square_mm
    marker_mm = args.marker_mm
    if marker_mm >= square_mm:
        raise SystemExit("--marker-mm must be smaller than --square-mm")

    board_mm = (cols * square_mm, rows * square_mm)
    avail = (PAGE_MM[0] - 2 * MARGIN_MM, PAGE_MM[1] - 2 * MARGIN_MM)
    if board_mm[0] > avail[0] or board_mm[1] > avail[1]:
        raise SystemExit(
            f"Board {board_mm[0]:.0f}x{board_mm[1]:.0f} mm doesn't fit on "
            f"letter portrait with {MARGIN_MM:.0f} mm margins "
            f"(printable {avail[0]:.0f}x{avail[1]:.0f} mm). "
            f"Reduce --cols/--rows or --square-mm."
        )

    px_per_mm = DPI / 25.4

    def mm_to_px(mm: float) -> int:
        return int(round(mm * px_per_mm))

    page_w, page_h = mm_to_px(PAGE_MM[0]), mm_to_px(PAGE_MM[1])
    board_w, board_h = mm_to_px(board_mm[0]), mm_to_px(board_mm[1])

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (cols, rows), square_mm / 1000.0, marker_mm / 1000.0, dictionary
    )
    board_img = board.generateImage((board_w, board_h))

    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    y0 = mm_to_px(MARGIN_MM)
    x0 = (page_w - board_w) // 2
    page[y0 : y0 + board_h, x0 : x0 + board_w] = board_img

    label_y = y0 + board_h + mm_to_px(10)
    label = (
        f"ChArUco {cols}x{rows} squares  square={square_mm:.1f}mm  "
        f"marker={marker_mm:.1f}mm"
    )
    cv2.putText(
        page, label, (mm_to_px(MARGIN_MM), label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2,
    )

    bar_y = label_y + mm_to_px(10)
    bar_x0 = mm_to_px(MARGIN_MM)
    bar_len_mm = 100.0
    bar_x1 = bar_x0 + mm_to_px(bar_len_mm)
    cv2.line(page, (bar_x0, bar_y), (bar_x1, bar_y), 0, 3)
    for mm in range(0, int(bar_len_mm) + 1, 10):
        x = bar_x0 + mm_to_px(mm)
        h = mm_to_px(4 if mm % 50 == 0 else 2)
        cv2.line(page, (x, bar_y - h), (x, bar_y + h), 0, 2)
    cv2.putText(
        page,
        "Verify: this bar should measure exactly 100mm after printing",
        (bar_x0, bar_y + mm_to_px(8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), page)
    print(f"Wrote {args.out} ({DPI} DPI, US Letter portrait)")
    print("Print at ACTUAL SIZE / 100% (uncheck 'fit to page'/'shrink to fit').")
    print("Verify the 100 mm scale bar with a ruler before calibrating.\n")
    print("Calibrate command:")
    print(
        f"  uv run python -m perception calibrate \\\n"
        f"    --cols {cols} --rows {rows} \\\n"
        f"    --square-mm {square_mm} --marker-mm {marker_mm} \\\n"
        f"    --stream-port 8080 --preview-dir captures"
    )


def _cmd_calibrate(args: argparse.Namespace) -> None:
    """Headless ChArUco intrinsic calibration → config/camera_intrinsics.json.

    No display required. Auto-captures whenever the board is detected with
    ≥ --min-corners ChArUco corners AND --cooldown-s has elapsed since the
    last capture. Move the board around in front of the camera; the counter
    in the terminal ticks up. Pass --preview-dir to also save each captured
    frame as a JPG for sanity-check after the fact.

    For a good calibration: vary distance (close + far), tilt the board in
    every direction, and make sure some captures push the board into the
    image corners — that's where lens distortion lives. RMS reprojection
    under ~0.5 px is good; over ~1.0 px means recapture with more variety.
    """
    square_m = args.square_mm / 1000.0
    marker_m = args.marker_mm / 1000.0
    if marker_m >= square_m:
        raise SystemExit("--marker-mm must be smaller than --square-mm")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (args.cols, args.rows), square_m, marker_m, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)

    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    obj_points_all: list[np.ndarray] = []
    img_points_all: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    last_capture_t = 0.0
    last_heartbeat_t = 0.0
    HEARTBEAT_S = 2.0

    log.info(
        "ChArUco %dx%d squares, %.1f mm square, %.1f mm marker. "
        "Target %d captures, ≥%d corners each.",
        args.cols, args.rows, args.square_mm, args.marker_mm,
        args.target_captures, args.min_corners,
    )
    log.info("Move the board around. Ctrl-C to abort early.")

    stream = None
    if args.stream_port:
        stream = MjpegServer(args.stream_port)
        stream.start()
    mount = CameraMount.load()  # only used to flip the stream for human eyes

    with Camera(args.device, args.width, args.height) as cam:
        while len(obj_points_all) < args.target_captures:
            frame = cam.read()
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
            ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(frame)
            n_ch = 0 if ch_corners is None else len(ch_corners)
            n_m = 0 if m_ids is None else len(m_ids)
            now = time.time()
            if now - last_heartbeat_t >= HEARTBEAT_S:
                log.info(
                    "[live] markers=%d charuco_corners=%d (need ≥%d to capture)",
                    n_m, n_ch, args.min_corners,
                )
                last_heartbeat_t = now
            if stream is not None:
                overlay = frame.copy()
                if n_m > 0:
                    cv2.aruco.drawDetectedMarkers(overlay, m_corners, m_ids)
                if n_ch > 0:
                    for pt in ch_corners.reshape(-1, 2).astype(int):
                        cv2.circle(overlay, tuple(pt), 4, (0, 255, 255), -1)
                cv2.putText(
                    overlay,
                    f"captures: {len(obj_points_all)}/{args.target_captures}  "
                    f"markers: {n_m}  charuco: {n_ch}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                if mount.rotated_180:
                    overlay = cv2.rotate(overlay, cv2.ROTATE_180)
                stream.push(overlay)
            ready = (
                n_ch >= args.min_corners
                and (now - last_capture_t) >= args.cooldown_s
            )
            if ready:
                obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
                if obj_pts is not None and len(obj_pts) >= args.min_corners:
                    obj_points_all.append(obj_pts)
                    img_points_all.append(img_pts)
                    last_capture_t = now
                    log.info(
                        "Captured %d/%d (%d corners)",
                        len(obj_points_all), args.target_captures, n_ch,
                    )
                    if args.preview_dir is not None:
                        cv2.imwrite(
                            str(args.preview_dir / f"cap_{len(obj_points_all):02d}.jpg"),
                            frame,
                        )

    log.info("Calibrating with %d views...", len(obj_points_all))
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points_all, img_points_all, image_size, None, None
    )
    log.info("Reprojection RMS: %.3f px", rms)
    if rms > 1.0:
        log.warning("RMS > 1.0 px — recapture with more pose variety.")
    CameraIntrinsics(
        matrix=K, distortion=dist.flatten(), image_size=image_size
    ).save()
    log.info("Saved → %s", DEFAULT_INTRINSICS_PATH)


def _cmd_teach_pose(args: argparse.Namespace) -> None:
    """Register or update a marker, and record its dock pose by demonstration.

    First time teaching a new marker:
        teach-pose --marker manipulator_far --tag-id 11 --size-m 0.095
    Re-teaching an existing marker (just refining the pose):
        teach-pose --marker manipulator_far

    Place the robot at the desired docking spot with the marker in view,
    then run. The script samples body-frame (forward, lateral) for
    `--duration-s`, writes the median into config/markers.json, and that's
    it. Re-runnable; safe to hand-edit the JSON.
    """
    from tasks import (
        DockPose,
        load_markers,
        register_marker,
        save_marker_pose,
    )

    if not DEFAULT_INTRINSICS_PATH.exists():
        raise SystemExit(
            f"No intrinsics at {DEFAULT_INTRINSICS_PATH}. Run `calibrate` first."
        )

    markers = load_markers()
    existing = markers.get(args.marker)
    tag_id = args.tag_id if args.tag_id is not None else (existing.tag_id if existing else None)
    size_m = args.size_m if args.size_m is not None else (existing.size_m if existing else None)
    if tag_id is None or size_m is None:
        raise SystemExit(
            f"Marker {args.marker!r} not registered. First-time teach needs "
            f"both --tag-id and --size-m. Known markers: {sorted(markers)}"
        )
    if existing is None:
        log.info("Registering new marker '%s' (tag=%d, %.0fmm)",
                 args.marker, tag_id, size_m * 1000)
    elif tag_id != existing.tag_id or abs(size_m - existing.size_m) > 1e-6:
        log.info(
            "Updating '%s': tag %d→%d, size %.0fmm→%.0fmm",
            args.marker, existing.tag_id, tag_id,
            existing.size_m * 1000, size_m * 1000,
        )
    m = register_marker(args.marker, tag_id, size_m)

    intrinsics = CameraIntrinsics.load()
    mount = CameraMount.load()
    detector = ArucoDetector(marker_size_m=m.size_m, intrinsics=intrinsics)

    log.info(
        "Teaching '%s' (tag=%d, %.0fmm). Hold the robot still at the desired "
        "dock pose. Sampling for %.1fs...",
        m.name, m.tag_id, m.size_m * 1000, args.duration_s,
    )

    samples: list[tuple[float, float]] = []
    deadline = time.time() + args.duration_s
    with Camera() as cam:
        while time.time() < deadline:
            frame = cam.read()
            target = next(
                (d for d in detector.detect(frame) if d.id == m.tag_id), None
            )
            if target is not None and target.tvec is not None:
                x_b, y_b, _ = camera_to_body(target.tvec, mount)
                samples.append((x_b, y_b))

    if len(samples) < args.min_samples:
        raise SystemExit(
            f"Only got {len(samples)} detections in {args.duration_s:.1f}s "
            f"(need ≥{args.min_samples}). Is the marker in view?"
        )

    arr = np.array(samples)
    fwd_med, lat_med = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
    fwd_std, lat_std = float(np.std(arr[:, 0])), float(np.std(arr[:, 1]))
    log.info(
        "Pose: forward=%.4fm  lateral=%.4fm  (std fwd=%.4fm lat=%.4fm, n=%d)",
        fwd_med, lat_med, fwd_std, lat_std, len(samples),
    )
    if max(fwd_std, lat_std) > 0.005:
        log.warning(
            "Std > 5 mm — the robot may be moving, or detection is noisy. "
            "Re-run with the chassis braced."
        )

    save_marker_pose(m.name, DockPose(forward_m=fwd_med, lateral_m=lat_med))
    log.info("Saved pose for '%s' → config/markers.json", m.name)


def _cmd_detect(args: argparse.Namespace) -> None:
    """Live ArUco detection. Logs IDs (and pose if intrinsics exist).

    For a preview, pass --stream-port and open http://<host>:PORT/ in a browser.
    """
    intrinsics = None
    if DEFAULT_INTRINSICS_PATH.exists():
        intrinsics = CameraIntrinsics.load()
        log.info("Loaded intrinsics from %s", DEFAULT_INTRINSICS_PATH)
    else:
        log.warning(
            "No intrinsics at %s — pose disabled. Run `calibrate` to enable.",
            DEFAULT_INTRINSICS_PATH,
        )

    detector = ArucoDetector(marker_size_m=args.marker_size_m, intrinsics=intrinsics)
    mount = CameraMount.load()

    stream = None
    if args.stream_port:
        stream = MjpegServer(args.stream_port)
        stream.start()

    last_log = 0.0
    with Camera(args.device) as cam:
        while True:
            frame = cam.read()
            detections = detector.detect(frame)
            now = time.time()
            if detections and now - last_log > 0.5:
                for d in detections:
                    if d.tvec is not None:
                        log.info(
                            "id=%d  tvec=(%.2f, %.2f, %.2f) m  dist=%.2f m",
                            d.id, *d.tvec, d.distance_m,
                        )
                    else:
                        cx, cy = d.centre_px
                        log.info("id=%d  centre=(%.0f, %.0f) px", d.id, cx, cy)
                last_log = now
            if stream is not None:
                detector.draw(frame, detections)
                display = cv2.rotate(frame, cv2.ROTATE_180) if mount.rotated_180 else frame
                stream.push(display)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(prog="perception")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Write station marker PNGs")
    g.add_argument("--out", type=Path, default=Path("markers"))
    g.add_argument("--pixels", type=int, default=600)
    g.set_defaults(func=_cmd_generate)

    b = sub.add_parser("make-board", help="Printable ChArUco board for US Letter")
    b.add_argument("--out", type=Path, default=Path("calibration_board.png"))
    b.add_argument("--cols", type=int, default=7, help="ChArUco squares across")
    b.add_argument("--rows", type=int, default=10, help="ChArUco squares down")
    b.add_argument("--square-mm", type=float, default=25.0)
    b.add_argument("--marker-mm", type=float, default=19.0)
    b.set_defaults(func=_cmd_make_board)

    c = sub.add_parser("calibrate", help="Headless ChArUco intrinsic calibration")
    c.add_argument("--device", default="/dev/video0")
    c.add_argument("--width", type=int, default=1920)
    c.add_argument("--height", type=int, default=1080)
    c.add_argument("--cols", type=int, default=7, help="ChArUco squares across")
    c.add_argument("--rows", type=int, default=10, help="ChArUco squares down")
    c.add_argument("--square-mm", type=float, default=25.0)
    c.add_argument("--marker-mm", type=float, default=19.0)
    c.add_argument("--target-captures", type=int, default=20)
    c.add_argument("--min-corners", type=int, default=6,
                   help="Minimum ChArUco corners required to accept a capture")
    c.add_argument("--cooldown-s", type=float, default=1.0,
                   help="Min seconds between captures (force pose variety)")
    c.add_argument("--preview-dir", type=Path, default=None,
                   help="If set, save each captured frame as JPG for review")
    c.add_argument("--stream-port", type=int, default=0,
                   help="If >0, serve MJPEG preview at http://<host>:PORT/")
    c.set_defaults(func=_cmd_calibrate)

    t = sub.add_parser("teach-pose", help="Register/teach a marker's dock pose")
    t.add_argument("--marker", required=True, help="Marker name")
    t.add_argument("--tag-id", type=int, default=None,
                   help="ArUco tag ID (required first time, ignored on re-teach)")
    t.add_argument("--size-m", type=float, default=None,
                   help="Printed marker edge in metres (required first time, ignored on re-teach)")
    t.add_argument("--duration-s", type=float, default=2.0)
    t.add_argument("--min-samples", type=int, default=20)
    t.set_defaults(func=_cmd_teach_pose)

    d = sub.add_parser("detect", help="Live ArUco detection")
    d.add_argument("--device", default="/dev/video0")
    d.add_argument("--marker-size-m", type=float, default=0.036)
    d.add_argument("--stream-port", type=int, default=0,
                   help="If >0, serve MJPEG preview at http://<host>:PORT/")
    d.set_defaults(func=_cmd_detect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
