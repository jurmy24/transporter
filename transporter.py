"""
Transporter: wheels-only fork of lerobot's LeKiwi.
Only the 3 omniwheel base motors (IDs 7, 8, 9). No arm, no cameras.
"""

import logging
import time
from functools import cached_property

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)


class Transporter(Robot):
    """Three-omniwheel mobile base. No arm, no cameras."""

    config_class = RobotConfig
    name = "transporter"

    def __init__(self, config: RobotConfig):
        super().__init__(config)
        self.config = config
        self.port = config.port
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.base_motors = list(self.bus.motors.keys())

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return dict.fromkeys(("x.vel", "y.vel", "theta.vel"), float)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(("x.vel", "y.vel", "theta.vel"), float)

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use existing calibration for '{self.id}', or 'c' to recalibrate: "
            )
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        homing_offsets = dict.fromkeys(self.base_motors, 0)
        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=0,
                range_max=4095,
            )
        self.bus.write_calibration(self.calibration)
        self._save_calibration()

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in self.base_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        self.bus.enable_torque()

    # -- Omniwheel kinematics --

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_int = int(round(degps * steps_per_deg))
        return max(-0x8000, min(0x7FFF, speed_int))

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        return raw_speed / (4096.0 / 360.0)

    def _body_to_wheel_raw(
        self, x: float, y: float, theta: float,
        wheel_radius: float = 0.05, base_radius: float = 0.125, max_raw: int = 3000,
    ) -> dict[str, int]:
        theta_rad = theta * (np.pi / 180.0)
        velocity = np.array([x, y, theta_rad])
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        wheel_degps = (m.dot(velocity) / wheel_radius) * (180.0 / np.pi)

        steps_per_deg = 4096.0 / 360.0
        max_computed = max(abs(d) * steps_per_deg for d in wheel_degps)
        if max_computed > max_raw:
            wheel_degps = wheel_degps * (max_raw / max_computed)

        raw = [self._degps_to_raw(d) for d in wheel_degps]
        return {
            "base_left_wheel": raw[0],
            "base_back_wheel": raw[1],
            "base_right_wheel": raw[2],
        }

    def _wheel_raw_to_body(
        self, left: int, back: int, right: int,
        wheel_radius: float = 0.05, base_radius: float = 0.125,
    ) -> dict[str, float]:
        wheel_degps = np.array([self._raw_to_degps(s) for s in (left, back, right)])
        wheel_linear = wheel_degps * (np.pi / 180.0) * wheel_radius
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        x, y, theta_rad = np.linalg.inv(m).dot(wheel_linear)
        return {"x.vel": x, "y.vel": y, "theta.vel": theta_rad * (180.0 / np.pi)}

    # -- Robot interface --

    @check_if_not_connected
    def get_observation(self):
        start = time.perf_counter()
        raw = self.bus.sync_read("Present_Velocity", self.base_motors)
        obs = self._wheel_raw_to_body(
            raw["base_left_wheel"], raw["base_back_wheel"], raw["base_right_wheel"],
        )
        logger.debug(f"{self} read state: {(time.perf_counter() - start) * 1e3:.1f}ms")
        return obs

    @check_if_not_connected
    def send_action(self, action) -> dict:
        x = action.get("x.vel", 0.0)
        y = action.get("y.vel", 0.0)
        theta = action.get("theta.vel", 0.0)
        wheel_cmds = self._body_to_wheel_raw(x, y, theta)
        self.bus.sync_write("Goal_Velocity", wheel_cmds)
        return {"x.vel": x, "y.vel": y, "theta.vel": theta}

    def stop_base(self):
        self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base stopped")

    @check_if_not_connected
    def disconnect(self):
        self.stop_base()
        if hasattr(self.config, "disable_torque_on_disconnect"):
            self.bus.disconnect(self.config.disable_torque_on_disconnect)
        else:
            self.bus.disconnect(True)
        logger.info(f"{self} disconnected.")
