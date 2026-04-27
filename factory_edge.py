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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

import paho.mqtt.client as mqtt
from pydantic import BaseModel, Field, ValidationError

from tasks import (
    DEFAULT_DISTANCE_TO_ASSEMBLER_M,
    DEFAULT_DISTANCE_TO_BASE_M,
    DEFAULT_DISTANCE_TO_DROP_OFF_M,
    STATION_TAG_IDS,
    go_to_assembler,
    go_to_drop_off,
    return_to_base,
    run_route,
)
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


class MachineResult(BaseModel):
    machine: str
    request_id: str
    event: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)


class PreflightCheck(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class PreflightReport(BaseModel):
    machine: str
    request_id: str
    checks: list[PreflightCheck]
    timestamp: datetime = Field(default_factory=_now)


# ── Transporter-specific command params ─────────────────────────────


class DispatchParams(BaseModel):
    """Params accepted by the `dispatch` command.

    The orchestrator is authoritative about location: each dispatch carries
    both `from_station` and `to_station`, so the transporter is stateless
    across legs and just looks up the route.
    """

    from_station: str
    to_station: str
    request_id: str = ""


_P = "factory"


def machine_state_topic(m: str) -> str:
    return f"{_P}/machines/{m}/state"


def machine_error_topic(m: str) -> str:
    return f"{_P}/machines/{m}/error"


def command_topic(m: str) -> str:
    return f"{_P}/commands/{m}"


def machine_result_topic(m: str) -> str:
    return f"{_P}/machines/{m}/results"


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


HandlerResult = tuple[bool, str, dict[str, Any]]
Handler = Callable[[dict[str, Any]], HandlerResult]

RECENT_REQUEST_IDS_MAX = 64


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

    # ── Handlers ─────────────────────────────────────────────────────

    def _start_leg(
        params: dict,
        runner: Callable[..., None],
        default_distance: float,
        leg_name: str,
    ) -> HandlerResult:
        """Shared body for dispatch + the three rpi_build legs.

        Each leg is just `IDLE → DELIVERING → run something → arrived`.
        The only thing that varies is which task function runs and what
        the default distance is, so we factor that out here.
        """
        if sm.state != S.IDLE:
            return (
                False,
                f"transporter is {sm.state.name}, cannot accept {leg_name}",
                {"state": sm.state.name},
            )
        distance = float(params.get("distance", default_distance))
        threading.Thread(
            target=run_leg,
            args=(runner, distance, leg_name),
            daemon=True,
        ).start()
        return True, f"{leg_name} started", {"distance": distance, "leg": leg_name}

    def h_dispatch(params: dict) -> HandlerResult:
        if sm.state != S.IDLE:
            return (
                False,
                f"transporter is {sm.state.name}, cannot accept dispatch",
                {"state": sm.state.name},
            )
        try:
            dp = DispatchParams.model_validate(params)
        except ValidationError as e:
            return False, f"invalid dispatch params: {e}", {"params": params}
        for name in (dp.from_station, dp.to_station):
            if name not in STATION_TAG_IDS:
                return False, f"unknown station: {name}", {"params": params}
        threading.Thread(
            target=run_delivery,
            args=(dp.from_station, dp.to_station),
            daemon=True,
        ).start()
        return (
            True,
            f"dispatched {dp.from_station} → {dp.to_station}",
            {"from": dp.from_station, "to": dp.to_station},
        )

    def h_go_to_assembler(params: dict) -> HandlerResult:
        return _start_leg(
            params, go_to_assembler, DEFAULT_DISTANCE_TO_ASSEMBLER_M, "go_to_assembler"
        )

    def h_go_to_drop_off(params: dict) -> HandlerResult:
        return _start_leg(
            params, go_to_drop_off, DEFAULT_DISTANCE_TO_DROP_OFF_M, "go_to_drop_off"
        )

    def h_return_to_base(params: dict) -> HandlerResult:
        return _start_leg(
            params, return_to_base, DEFAULT_DISTANCE_TO_BASE_M, "return_to_base"
        )

    def h_recover(params: dict) -> HandlerResult:
        if sm.state != S.ERROR:
            return (
                False,
                f"transporter is {sm.state.name}, not ERROR — ignoring recover",
                {"state": sm.state.name},
            )
        threading.Thread(target=run_recovery, daemon=True).start()
        return True, "recovery started", {}

    def h_stop(_params: dict) -> HandlerResult:
        try:
            robot.stop_base()
        except Exception as e:
            return False, f"stop_base failed: {e}", {}
        return True, "base stopped", {"state": sm.state.name}

    def h_pause(_params: dict) -> HandlerResult:
        """Stop-and-idle: halt the base and nudge the state machine to IDLE."""
        try:
            robot.stop_base()
        except Exception as e:
            log.warning("stop_base failed during pause: %s", e)
        try:
            if sm.state == S.DELIVERING:
                sm.send("arrived")
            elif sm.state == S.ERROR:
                sm.send("recover")
        except Exception as e:
            log.warning("pause transition failed: %s", e)
        return True, "paused", {"final_state": sm.state.name}

    def h_resume(_params: dict) -> HandlerResult:
        return (
            True,
            f"nothing to resume (state={sm.state.name})",
            {"state": sm.state.name},
        )

    def h_get_odometry(_params: dict) -> HandlerResult:
        try:
            obs = robot.get_observation()
        except Exception as e:
            return False, f"get_observation failed: {e}", {}
        # Observations can contain numpy types; coerce to plain floats.
        safe: dict[str, Any] = {}
        if isinstance(obs, dict):
            for k, v in obs.items():
                try:
                    safe[str(k)] = float(v)
                except (TypeError, ValueError):
                    safe[str(k)] = str(v)
        return True, "odometry", safe

    def h_preflight(params: dict) -> HandlerResult:
        """Minimal transporter preflight — standardises the verb across
        machines so Clippy can call check_readiness uniformly."""
        checks: list[dict[str, Any]] = []
        connected = False
        try:
            connected = bool(robot.is_connected)
        except Exception:
            connected = False
        checks.append(
            {
                "name": "transporter_connected",
                "ok": connected,
                "detail": ""
                if connected
                else "Motor bus is not connected. Plug the transporter in.",
                "data": {"connected": connected},
            }
        )
        idle = sm.state == S.IDLE
        checks.append(
            {
                "name": "transporter_idle",
                "ok": idle,
                "detail": "" if idle else f"Transporter is {sm.state.name}, not IDLE.",
                "data": {"state": sm.state.name},
            }
        )
        report = PreflightReport(
            machine=MACHINE_NAME,
            request_id="",
            checks=[PreflightCheck.model_validate(c) for c in checks],
        )
        return (
            all(c.ok for c in report.checks),
            "preflight complete",
            report.model_dump(mode="json"),
        )

    HANDLERS: dict[str, Handler] = {
        "dispatch": h_dispatch,
        "go_to_assembler": h_go_to_assembler,
        "go_to_drop_off": h_go_to_drop_off,
        "return_to_base": h_return_to_base,
        "recover": h_recover,
        "stop": h_stop,
        "pause": h_pause,
        "resume": h_resume,
        "get_odometry": h_get_odometry,
        "preflight": h_preflight,
    }

    recent_req_ids: deque[str] = deque(maxlen=RECENT_REQUEST_IDS_MAX)

    def publish_result(
        req_id: str, event: str, ok: bool, detail: str, data: dict
    ) -> None:
        if not req_id:
            return
        mqtt_client.publish_model(
            machine_result_topic(MACHINE_NAME),
            MachineResult(
                machine=MACHINE_NAME,
                request_id=req_id,
                event=event,
                ok=ok,
                detail=detail,
                data=data,
            ),
        )

    def on_command(_topic: str, data: dict) -> None:
        try:
            cmd = Command.model_validate(data)
        except Exception:
            log.warning("Invalid command: %s", data)
            return
        handler = HANDLERS.get(cmd.event)
        if handler is None:
            log.warning("Unknown command event: %s", cmd.event)
            return
        req_id = str(cmd.params.get("request_id") or "")
        if req_id and req_id in recent_req_ids:
            log.info("Dropping duplicate request_id=%s", req_id)
            return
        if req_id:
            recent_req_ids.append(req_id)

        def run() -> None:
            try:
                ok, detail, payload = handler(cmd.params)
            except Exception as e:
                log.exception("Handler %s crashed", cmd.event)
                ok, detail, payload = False, f"handler crashed: {e}", {}
            publish_result(req_id, cmd.event, ok, detail, payload)

        threading.Thread(target=run, daemon=True).start()

    # ── Long-running workers ─────────────────────────────────────────

    def run_delivery(from_station: str, to_station: str) -> None:
        try:
            sm.send("dispatch")
            run_route(robot, from_station, to_station)
            sm.send("arrived")
        except Exception as e:
            log.error("Delivery %s → %s failed: %s", from_station, to_station, e)
            try:
                robot.stop_base()
            except Exception:
                pass
            if sm.state == S.DELIVERING:
                sm.send("fault")
                publish_error(
                    "drive_failed",
                    f"route {from_station} → {to_station}: {e}",
                )

    def run_leg(
        runner: Callable[..., None],
        distance: float,
        leg_name: str,
    ) -> None:
        """Drive one leg: enter DELIVERING, run the task, arrive (or fault)."""
        try:
            sm.send("dispatch")
            runner(robot, distance)
            sm.send("arrived")
        except Exception as e:
            log.error("Leg %s failed: %s", leg_name, e)
            try:
                robot.stop_base()
            except Exception:
                pass
            if sm.state == S.DELIVERING:
                sm.send("fault")
                publish_error("drive_failed", f"{leg_name}: {e}")

    def run_recovery() -> None:
        try:
            robot.stop_base()
        except Exception:
            pass
        try:
            sm.send("recover")
        except Exception as e:
            log.error("Recovery transition failed: %s", e)

    # ── Main loop ────────────────────────────────────────────────────

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
