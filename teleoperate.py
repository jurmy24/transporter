"""
Run on the laptop. Drives the wheels over ZMQ using keyboard input.

Usage:
    python teleoperate.py
    python teleoperate.py --ip 172.20.10.2 --id biden_kiwi
"""

import argparse
import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import (
    KeyboardTeleop,
    KeyboardTeleopConfig,
)
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def main():
    parser = argparse.ArgumentParser(
        description="Keyboard teleoperation for LeKiwi wheels"
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="172.20.10.2",
        help="Remote IP of the Pi running host.py",
    )
    parser.add_argument(
        "--id", type=str, default="biden_kiwi", help="Robot ID (must match host.py)"
    )
    parser.add_argument(
        "--keyboard_id",
        type=str,
        default="my_laptop_keyboard",
        help="Keyboard teleop ID",
    )
    args = parser.parse_args()

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.ip, id=args.id))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id=args.keyboard_id))

    robot.connect()
    keyboard.connect()

    if not robot.is_connected or not keyboard.is_connected:
        raise RuntimeError("Failed to connect to robot or keyboard")

    print("Teleop active. Use arrow keys / WASD to drive.")
    try:
        while True:
            t0 = time.perf_counter()
            base_action = robot._from_keyboard_to_base_action(keyboard.get_action())
            if base_action:
                robot.send_action(base_action)
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.disconnect()
        keyboard.disconnect()


if __name__ == "__main__":
    main()
