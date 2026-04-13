"""
Run on the Pi. Starts the lerobot ZMQ server that bridges remote commands to the Feetech motors.

Usage:
    python host.py
    python host.py --robot.port=/dev/ttyACM0 --robot.id=biden_kiwi
"""

import sys

from lerobot.robots.lekiwi.lekiwi_host import main

if __name__ == "__main__":
    if not any("robot.port" in arg for arg in sys.argv):
        sys.argv.append("--robot.port=/dev/ttyACM0")
    if not any("robot.id" in arg for arg in sys.argv):
        sys.argv.append("--robot.id=biden_kiwi")
    if not any("host.connection_time_s" in arg for arg in sys.argv):
        sys.argv.append("--host.connection_time_s=600")

    main()
