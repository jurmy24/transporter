"""
Run on the Pi. ZMQ server that receives velocity commands and drives the wheels.

Usage:
    python host.py
    python host.py --port /dev/ttyACM0 --id biden_kiwi --connection-time 600
"""

import argparse
import json
import logging
import time

import zmq

from transporter import Transporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Transporter host (Pi-side)")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port for the motor bus")
    parser.add_argument("--id", type=str, default="biden_kiwi", help="Robot ID")
    parser.add_argument("--zmq-cmd-port", type=int, default=5555, help="ZMQ port for receiving commands")
    parser.add_argument("--zmq-obs-port", type=int, default=5556, help="ZMQ port for sending observations")
    parser.add_argument("--connection-time", type=int, default=600, help="Max session duration (seconds)")
    parser.add_argument("--watchdog-ms", type=int, default=500, help="Watchdog timeout (ms)")
    parser.add_argument("--max-freq", type=int, default=30, help="Max loop frequency (Hz)")
    args = parser.parse_args()

    # Robot
    from lerobot.robots.config import RobotConfig
    from dataclasses import dataclass

    @dataclass
    class TransporterConfig(RobotConfig):
        port: str = args.port
        disable_torque_on_disconnect: bool = True

    config = TransporterConfig(id=args.id, port=args.port)
    robot = Transporter(config)

    logger.info("Connecting to motors...")
    robot.connect()

    # ZMQ sockets
    ctx = zmq.Context()
    cmd_socket = ctx.socket(zmq.PULL)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.bind(f"tcp://*:{args.zmq_cmd_port}")

    obs_socket = ctx.socket(zmq.PUSH)
    obs_socket.setsockopt(zmq.CONFLATE, 1)
    obs_socket.bind(f"tcp://*:{args.zmq_obs_port}")

    logger.info("Waiting for commands...")
    last_cmd_time = time.time()
    watchdog_active = False

    try:
        start = time.perf_counter()
        while (time.perf_counter() - start) < args.connection_time:
            loop_start = time.time()

            # Receive command
            try:
                msg = cmd_socket.recv_string(zmq.NOBLOCK)
                action = json.loads(msg)
                robot.send_action(action)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"Command error: {e}")

            # Watchdog
            if (time.time() - last_cmd_time > args.watchdog_ms / 1000) and not watchdog_active:
                logger.warning("Watchdog: no command received, stopping base.")
                watchdog_active = True
                robot.stop_base()

            # Send observation
            obs = robot.get_observation()
            try:
                obs_socket.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            elapsed = time.time() - loop_start
            time.sleep(max(1.0 / args.max_freq - elapsed, 0))

        logger.info("Session time reached.")
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        robot.disconnect()
        obs_socket.close()
        cmd_socket.close()
        ctx.term()
        logger.info("Shut down.")


if __name__ == "__main__":
    main()
