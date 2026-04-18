"""Transporter factory edge — MQTT bridge + state machine.

Runs on the Raspberry Pi. Replaces host.py as the entry point.
Publishes state to the MQTT broker and receives commands from the orchestrator.

Self-contained: only depends on paho-mqtt, pydantic, and lerobot[feetech].

Usage:
    python factory_edge.py --broker 192.168.1.100 --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

import paho.mqtt.client as mqtt
from pydantic import BaseModel, Field

from tasks import drive_forward
from transporter import Transporter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("transporter.edge")


# ── Vendored shared code ────────────────────────────────────────────
# TODO: Replace with `from paperclip_shared import ...` once factory/src/shared
# is published as a pip-installable package. Until then, this is inlined so the
# edge file runs standalone on the Pi without access to the factory repo.


def _now() -> datetime:
    return datetime.now(UTC)


class MachineState(BaseModel):
    machine: str
    state: str
    previous_state: str | None = None
    event: str | None = None
    task_complete: bool = False
    timestamp: datetime = Field(default_factory=_now)


class MachineError(BaseModel):
    machine: str
    error: str
    state_at_fault: str
    context: str = ""
    timestamp: datetime = Field(default_factory=_now)


class Command(BaseModel):
    event: str
    params: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)


_P = "factory"


def machine_state_topic(m: str) -> str:
    return f"{_P}/machines/{m}/state"


def machine_error_topic(m: str) -> str:
    return f"{_P}/machines/{m}/error"


def command_topic(m: str) -> str:
    return f"{_P}/commands/{m}"


MessageCallback = Callable[[str, dict[str, Any]], None]


class MqttClient:
    def __init__(
        self, client_id: str, host: str = "localhost", port: int = 1883
    ) -> None:
        self._host, self._port = host, port
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._connected = threading.Event()
        self._subs: dict[str, MessageCallback] = {}

    def connect(self) -> None:
        self._connected.clear()
        self._client.connect(self._host, self._port)

    def wait_for_connection(self, timeout: float = 5.0) -> bool:
        return self._connected.wait(timeout=timeout)

    def loop_start(self) -> None:
        self._client.loop_start()

    def loop_stop(self) -> None:
        self._client.loop_stop()

    def disconnect(self) -> None:
        self._client.disconnect()

    def publish_model(
        self, topic: str, model: BaseModel, *, retain: bool = False
    ) -> None:
        self._client.publish(topic, model.model_dump_json(), retain=retain)

    def subscribe(self, topic: str, callback: MessageCallback) -> None:
        self._subs[topic] = callback
        if self._client.is_connected():
            self._client.subscribe(topic)

    def _on_connect(self, *args: Any, **kw: Any) -> None:
        rc = args[3] if len(args) > 3 else kw.get("rc", 0)
        if hasattr(rc, "value"):
            rc = rc.value
        if rc == 0:
            self._connected.set()
            for t in self._subs:
                self._client.subscribe(t)

    def _on_message(self, _c: Any, _u: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            data = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        for t, cb in self._subs.items():
            if mqtt.topic_matches_sub(t, msg.topic):
                cb(msg.topic, data)

    def _on_disconnect(self, *_a: Any, **_kw: Any) -> None:
        self._connected.clear()


class InvalidTransition(Exception):
    pass


class StateMachine:
    def __init__(
        self,
        states: type[Enum],
        initial: Enum,
        transitions: dict,
        on_transition: Callable[[Enum, str, Enum], Any] | None = None,
    ) -> None:
        self._state = initial
        self._transitions = transitions
        self.on_transition = on_transition

    @property
    def state(self) -> Enum:
        return self._state

    def send(self, event: str) -> Enum:
        allowed = self._transitions.get(self._state, {})
        if event not in allowed:
            raise InvalidTransition(
                f"No '{event}' from {self._state.name}. Allowed: {list(allowed)}"
            )
        old = self._state
        self._state = allowed[event]
        log.info("%s --%s--> %s", old.name, event, self._state.name)
        if self.on_transition:
            self.on_transition(old, event, self._state)
        return self._state


# ── Transporter states & transitions ────────────────────────────────

MACHINE_NAME = "transporter"


class S(Enum):
    IDLE = auto()
    DELIVERING = auto()
    ERROR = auto()


TRANSITIONS = {
    S.IDLE: {"dispatch": S.DELIVERING},
    S.DELIVERING: {
        "arrived": S.IDLE,
        "fault": S.ERROR,
    },
    S.ERROR: {"recover": S.IDLE},
}


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Transporter factory edge (MQTT bridge)"
    )
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument(
        "--port", default="/dev/ttyACM0", help="Serial port for motor bus"
    )
    parser.add_argument("--id", default="transporter", help="Robot ID")
    args = parser.parse_args()

    from lerobot.robots.config import RobotConfig

    @dataclass
    class TransporterConfig(RobotConfig):
        port: str = args.port
        disable_torque_on_disconnect: bool = True

    robot = Transporter(TransporterConfig(id=args.id, port=args.port))
    log.info("Connecting to motors on %s...", args.port)
    robot.connect()

    mqtt_client = MqttClient(
        client_id="transporter-edge", host=args.broker, port=args.mqtt_port
    )

    def publish_state(old: Enum, event: str, new: Enum):
        mqtt_client.publish_model(
            machine_state_topic(MACHINE_NAME),
            MachineState(
                machine=MACHINE_NAME,
                state=new.name,
                previous_state=old.name,
                event=event,
                task_complete=(event == "arrived"),
            ),
            retain=True,
        )

    sm = StateMachine(S, S.IDLE, TRANSITIONS, on_transition=publish_state)

    def publish_error(error: str, context: str = ""):
        mqtt_client.publish_model(
            machine_error_topic(MACHINE_NAME),
            MachineError(
                machine=MACHINE_NAME,
                error=error,
                state_at_fault=sm.state.name,
                context=context,
            ),
        )

    last_distance: float = 0.5

    def on_command(_topic: str, data: dict):
        nonlocal last_distance
        try:
            cmd = Command.model_validate(data)
        except Exception:
            log.warning("Invalid command: %s", data)
            return
        if cmd.event == "dispatch" and sm.state == S.IDLE:
            last_distance = cmd.params.get("distance", 0.5)
            threading.Thread(target=run_delivery, daemon=True).start()
        elif cmd.event == "recover" and sm.state == S.ERROR:
            threading.Thread(target=run_recovery, daemon=True).start()

    def run_delivery():
        try:
            sm.send("dispatch")
            drive_forward(robot, last_distance)
            sm.send("arrived")
        except Exception as e:
            log.error("Delivery failed: %s", e)
            try:
                robot.stop_base()
            except Exception:
                pass
            if sm.state == S.DELIVERING:
                sm.send("fault")
                publish_error("drive_failed", str(e))

    def run_recovery():
        try:
            robot.stop_base()
        except Exception:
            pass
        try:
            sm.send("recover")
        except Exception as e:
            log.error("Recovery transition failed: %s", e)

    mqtt_client.connect()
    mqtt_client.loop_start()
    if not mqtt_client.wait_for_connection(timeout=10):
        log.error(
            "Could not connect to MQTT broker at %s:%d", args.broker, args.mqtt_port
        )
        robot.disconnect()
        return

    mqtt_client.subscribe(command_topic(MACHINE_NAME), on_command)
    mqtt_client.publish_model(
        machine_state_topic(MACHINE_NAME),
        MachineState(machine=MACHINE_NAME, state=sm.state.name),
        retain=True,
    )
    log.info(
        "Transporter edge running. State: %s. Waiting for commands...", sm.state.name
    )

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())

    try:
        stop.wait()
        log.info("Shutting down.")
    finally:
        robot.disconnect()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    main()
