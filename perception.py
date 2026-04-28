"""Camera + ArUco perception, plus dev CLI.

Library surface:
    Camera                  — context-manager wrapper around cv2.VideoCapture.
    CameraIntrinsics        — load/save 3x3 K and distortion coefficients.
    ArucoDetector           — DICT_4X4_50 detection; pose if intrinsics given.
    MarkerDetection         — id, image-frame corners, optional rvec/tvec.

Pose convention: tvec is (x, y, z) in metres in the OpenCV camera frame
(x right, y down, z forward). Convert to body frame at the docking call
site once the camera mount transform is known.

CLI (run from the repo root):
    python -m perception generate            # PNGs of station markers
    python -m perception make-board          # printable ChArUco for US Letter
    python -m perception calibrate           # headless ChArUco calibration
    python -m perception detect              # live detection (browser preview)

Calibration intrinsics are stored at config/camera_intrinsics.json so that
hardware-specific config lives separately from source.
"""

from __future__ import annotations

import argparse
import json
import logging
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
        marker_size_m: float = 0.05,
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
    PAGE_SIZES_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
    PAGE_MM = PAGE_SIZES_MM[args.paper]
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
    # Use integer pixels-per-square so generateImage doesn't fail an internal
    # ROI assertion when (board_w, board_h) doesn't divide evenly into cells.
    cell_px = mm_to_px(square_mm)
    board_w, board_h = cols * cell_px, rows * cell_px

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (cols, rows), square_mm / 1000.0, marker_mm / 1000.0, dictionary
    )
    board_img = board.generateImage((board_w, board_h), marginSize=0, borderBits=1)

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
    if args.out.suffix.lower() == ".pdf":
        from PIL import Image

        Image.fromarray(page, mode="L").save(
            str(args.out), "PDF", resolution=float(DPI)
        )
    else:
        cv2.imwrite(str(args.out), page)
    print(f"Wrote {args.out} ({DPI} DPI, {args.paper.upper()} portrait)")
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
                stream.push(frame)


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
    b.add_argument("--out", type=Path, default=Path("calibration_board.pdf"))
    b.add_argument("--paper", choices=["a4", "letter"], default="a4")
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

    d = sub.add_parser("detect", help="Live ArUco detection")
    d.add_argument("--device", default="/dev/video0")
    d.add_argument("--marker-size-m", type=float, default=0.05)
    d.add_argument("--stream-port", type=int, default=0,
                   help="If >0, serve MJPEG preview at http://<host>:PORT/")
    d.set_defaults(func=_cmd_detect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
