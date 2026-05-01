"""Transporter dashboard — web UI for control and testing.

Bridges a browser ↔ the MQTT broker so you can drive the transporter
without writing CLI commands. Runs anywhere that can reach the broker;
typically on the same Pi as ``factory_edge.py`` or on your laptop.

What it does:
  • Subscribes to ``factory/machines/transporter/{state,error,results}``
    and streams updates to the browser via Server-Sent Events.
  • Publishes commands on ``factory/commands/transporter`` with a
    fresh ``request_id`` so results can be correlated back.
  • Serves a single-page UI with quick actions, dispatch, test legs,
    diagnostics, an event log, and an embedded MJPEG camera preview.

Usage:
    python dashboard.py --broker 10.8.210.18 --http-port 8000
    # then open http://<this-host>:8000/

The MJPEG preview is served by ``factory_edge.py`` itself when it is
launched with ``--stream-port``. Point ``--stream-url`` at that.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify, render_template_string, request

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("transporter.dashboard")


# Topic conventions match factory_edge.py. Hard-coded here on purpose so the
# dashboard can run on a host that doesn't have the rest of the project.
MACHINE = "transporter"
TOPIC_STATE = f"factory/machines/{MACHINE}/state"
TOPIC_ERROR = f"factory/machines/{MACHINE}/error"
TOPIC_RESULTS = f"factory/machines/{MACHINE}/results"
TOPIC_COMMAND = f"factory/commands/{MACHINE}"

# Stations & test legs the UI exposes. Keep this list in sync with
# tasks.STATIONS / the leg handlers in factory_edge.py.
STATIONS = ("picker", "manipulator", "delivery")
LEG_COMMANDS = ("go_to_assembler", "go_to_drop_off", "return_to_base")

EVENT_HISTORY = 200  # rolling event log size


# ── Bridge: MQTT thread + SSE broadcaster ───────────────────────────


@dataclass
class DashboardState:
    """In-memory mirror of everything the UI needs to render.

    A single lock guards the whole struct because writes are infrequent
    (one MQTT message at a time) and reads are cheap. The event deque
    is what /api/state hands to a fresh browser tab so it can backfill
    history without waiting for the next live event.
    """
    mqtt_connected: bool = False
    last_state: dict[str, Any] | None = None
    last_error: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    last_odometry: dict[str, float] | None = None
    last_preflight: dict[str, Any] | None = None
    events: deque = field(default_factory=lambda: deque(maxlen=EVENT_HISTORY))


class Broadcaster:
    """Fan-out 'event' messages to every connected SSE client.

    Each browser tab gets its own bounded queue. Slow clients drop
    messages rather than backing up the producer; this is a
    monitoring UI, not a transactional log, so loss is acceptable.
    """

    def __init__(self) -> None:
        self._subs: list[queue.Queue[str]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=64)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, payload: dict[str, Any]) -> None:
        msg = json.dumps(payload, default=str)
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # drop for slow clients; UI will backfill on reconnect


class MqttBridge:
    """Wraps paho-mqtt with the dashboard's view of the world.

    Owns the connection, subscribes to the three transporter topics,
    updates ``DashboardState``, and pushes a structured event onto
    the broadcaster every time something changes.
    """

    def __init__(
        self,
        host: str,
        port: int,
        state: DashboardState,
        broadcaster: Broadcaster,
    ) -> None:
        self._host, self._port = host, port
        self._state = state
        self._broadcaster = broadcaster
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"dashboard-{uuid.uuid4().hex[:8]}"
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._lock = threading.Lock()

    def start(self) -> None:
        # Loop forever — paho's reconnect_delay handles retries while the
        # broker is unreachable, so a stale broker IP doesn't crash the UI.
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        try:
            self._client.connect_async(self._host, self._port)
        except Exception as e:
            log.warning("MQTT connect_async raised %s; loop will keep retrying", e)
        self._client.loop_start()

    def stop(self) -> None:
        try:
            self._client.disconnect()
        finally:
            self._client.loop_stop()

    def publish_command(self, event: str, params: dict[str, Any]) -> str:
        """Publish a command and return the assigned request_id."""
        req_id = uuid.uuid4().hex
        # Always carry request_id inside params so the edge can echo it on
        # the results topic; that's the only correlation handle we have.
        merged = {**params, "request_id": req_id}
        payload = {
            "event": event,
            "params": merged,
            "timestamp": _now_iso(),
        }
        self._client.publish(TOPIC_COMMAND, json.dumps(payload))
        self._push_event(
            kind="command_sent",
            data={"event": event, "params": merged, "request_id": req_id},
        )
        return req_id

    # ── paho callbacks ──────────────────────────────────────────────

    def _on_connect(self, client: mqtt.Client, *_: Any, **__: Any) -> None:
        log.info("MQTT connected to %s:%d", self._host, self._port)
        with self._lock:
            self._state.mqtt_connected = True
        for topic in (TOPIC_STATE, TOPIC_ERROR, TOPIC_RESULTS):
            client.subscribe(topic)
        self._push_event(kind="mqtt_connected", data={"host": self._host, "port": self._port})

    def _on_disconnect(self, *_: Any, **__: Any) -> None:
        log.warning("MQTT disconnected")
        with self._lock:
            self._state.mqtt_connected = False
        self._push_event(kind="mqtt_disconnected", data={})

    def _on_message(self, _c: Any, _u: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            data = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning("Bad payload on %s: %s", msg.topic, e)
            return

        with self._lock:
            if msg.topic == TOPIC_STATE:
                self._state.last_state = data
                self._push_event(kind="state", data=data)
            elif msg.topic == TOPIC_ERROR:
                self._state.last_error = data
                self._push_event(kind="error", data=data)
            elif msg.topic == TOPIC_RESULTS:
                self._state.last_result = data
                # Cache the typed payloads we render specially so
                # /api/state can hand them to a freshly-loaded UI.
                event = data.get("event")
                payload = data.get("data") or {}
                if event == "get_odometry" and data.get("ok"):
                    self._state.last_odometry = {
                        k: float(v)
                        for k, v in payload.items()
                        if isinstance(v, (int, float))
                    }
                elif event == "preflight":
                    self._state.last_preflight = payload
                self._push_event(kind="result", data=data)

    # ── helpers ─────────────────────────────────────────────────────

    def _push_event(self, *, kind: str, data: dict[str, Any]) -> None:
        evt = {"kind": kind, "ts": _now_iso(), "data": data}
        self._state.events.append(evt)
        self._broadcaster.publish(evt)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


# ── Flask app ───────────────────────────────────────────────────────


def create_app(
    bridge: MqttBridge,
    state: DashboardState,
    broadcaster: Broadcaster,
    *,
    stream_url: str | None,
    stream_port: int,
    stream_enabled: bool,
    broker_label: str,
) -> Flask:
    app = Flask(__name__)
    # The dashboard intentionally has no auth — it's a control panel for a
    # single robot on a trusted lab network. Don't expose it to the public.

    @app.route("/")
    def index() -> str:
        return render_template_string(
            INDEX_HTML,
            stations=STATIONS,
            leg_commands=LEG_COMMANDS,
            stream_url=stream_url or "",
            stream_port=stream_port,
            stream_enabled=stream_enabled,
            broker_label=broker_label,
        )

    @app.route("/api/state")
    def api_state() -> Response:
        # Snapshot under the same lock the bridge uses, so the UI can't
        # observe a partial update where last_state matches an event that
        # the events deque doesn't have yet.
        with bridge._lock:
            payload = {
                "mqtt_connected": state.mqtt_connected,
                "broker": broker_label,
                "stream_url": stream_url or "",
                "stream_port": stream_port,
                "stream_enabled": stream_enabled,
                "stations": list(STATIONS),
                "leg_commands": list(LEG_COMMANDS),
                "last_state": state.last_state,
                "last_error": state.last_error,
                "last_result": state.last_result,
                "last_odometry": state.last_odometry,
                "last_preflight": state.last_preflight,
                "events": list(state.events),
            }
        return jsonify(payload)

    @app.route("/api/command", methods=["POST"])
    def api_command() -> Response:
        body = request.get_json(silent=True) or {}
        event = body.get("event")
        params = body.get("params") or {}
        if not isinstance(event, str) or not event:
            return jsonify({"ok": False, "error": "missing 'event'"}), 400
        if not isinstance(params, dict):
            return jsonify({"ok": False, "error": "'params' must be an object"}), 400
        if not state.mqtt_connected:
            return jsonify({"ok": False, "error": "MQTT not connected"}), 503
        req_id = bridge.publish_command(event, params)
        return jsonify({"ok": True, "request_id": req_id})

    @app.route("/api/events")
    def api_events() -> Response:
        # SSE stream. We hold the request open and yield messages from a
        # per-client queue that the broadcaster fills.
        q = broadcaster.subscribe()

        def gen():
            try:
                yield ":connected\n\n"  # SSE comment so EventSource fires onopen
                while True:
                    try:
                        msg = q.get(timeout=15.0)
                        yield f"data: {msg}\n\n"
                    except queue.Empty:
                        # Heartbeat: keeps proxies from dropping the connection.
                        yield ":keepalive\n\n"
            finally:
                broadcaster.unsubscribe(q)

        return Response(gen(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable buffering on nginx, if any
        })

    return app


# ── HTML / CSS / JS ─────────────────────────────────────────────────
# Inlined so the dashboard ships as a single file. Jinja2 only fills in
# the station list, leg names, and the MJPEG stream URL.

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Transporter Dashboard</title>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --panel-2: #1f2630;
      --border: #2a3340;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --ok: #3fb950;
      --warn: #d29922;
      --err: #f85149;
      --idle: #58a6ff;
      --delivering: #d29922;
      --error: #f85149;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0;
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
                   "Helvetica Neue", Arial, sans-serif;
      font-size: 14px; line-height: 1.4;
    }
    header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 10;
    }
    header h1 { margin: 0; font-size: 18px; font-weight: 600; }
    header h1 .sub { color: var(--muted); font-weight: 400; margin-left: 8px; font-size: 13px; }
    .pills { display: flex; gap: 8px; align-items: center; }
    .pill {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 10px; border-radius: 999px;
      background: var(--panel-2); border: 1px solid var(--border);
      font-size: 12px; font-weight: 500;
    }
    .pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
    .pill.ok .dot { background: var(--ok); box-shadow: 0 0 0 3px rgba(63,185,80,0.15); }
    .pill.warn .dot { background: var(--warn); box-shadow: 0 0 0 3px rgba(210,153,34,0.15); }
    .pill.err .dot { background: var(--err); box-shadow: 0 0 0 3px rgba(248,81,73,0.15); }
    .pill.idle .dot { background: var(--idle); box-shadow: 0 0 0 3px rgba(88,166,255,0.15); }

    main {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
      gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto;
    }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }

    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 10px; padding: 14px 16px; margin-bottom: 16px;
    }
    .card h2 {
      margin: 0 0 10px 0; font-size: 14px; font-weight: 600;
      color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
    }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .row + .row { margin-top: 8px; }
    label { color: var(--muted); font-size: 12px; }

    button, select, input[type="number"], input[type="text"] {
      background: var(--panel-2); color: var(--text);
      border: 1px solid var(--border); border-radius: 6px;
      padding: 8px 12px; font-size: 13px;
      font-family: inherit;
    }
    button { cursor: pointer; transition: background 0.12s, border-color 0.12s; }
    button:hover { background: #2a3340; border-color: #3a4654; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.primary { background: var(--accent); border-color: var(--accent); color: #0d1117; font-weight: 600; }
    button.primary:hover { background: #79b8ff; }
    button.danger  { background: var(--err); border-color: var(--err); color: #fff; font-weight: 600; }
    button.danger:hover { background: #ff6a61; }
    button.warn  { background: var(--warn); border-color: var(--warn); color: #0d1117; font-weight: 600; }

    .stop-big { font-size: 18px; padding: 14px 20px; width: 100%; }

    pre, code {
      background: #0a0d12; border: 1px solid var(--border); border-radius: 6px;
      padding: 8px 10px; font-size: 12px; color: #c9d1d9;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap; word-break: break-all; margin: 0;
    }

    .kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; font-size: 13px; }
    .kv dt { color: var(--muted); }
    .kv dd { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }

    #log {
      max-height: 360px; overflow: auto;
      background: #0a0d12; border: 1px solid var(--border); border-radius: 6px;
      padding: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    #log .entry { padding: 2px 4px; border-bottom: 1px solid #14181f; display: grid;
                  grid-template-columns: 88px 90px 1fr; gap: 8px; align-items: start; }
    #log .entry:last-child { border-bottom: none; }
    #log .ts { color: var(--muted); }
    #log .kind { font-weight: 600; }
    #log .kind.state { color: var(--idle); }
    #log .kind.error { color: var(--err); }
    #log .kind.result { color: var(--ok); }
    #log .kind.command_sent { color: var(--accent); }
    #log .kind.mqtt_connected { color: var(--ok); }
    #log .kind.mqtt_disconnected { color: var(--err); }
    #log .body { color: #c9d1d9; word-break: break-word; }

    .stream-wrap {
      background: #000; border: 1px solid var(--border); border-radius: 6px;
      overflow: hidden; min-height: 240px; display: flex; align-items: center; justify-content: center;
      color: var(--muted); font-size: 13px;
    }
    .stream-wrap img { width: 100%; display: block; }
    .stream-wrap .placeholder { padding: 40px 20px; text-align: center; max-width: 380px; }
    .stream-wrap .placeholder a { color: var(--accent); }
    .reload-btn {
      float: right; padding: 2px 8px; font-size: 13px;
      background: var(--panel-2); border: 1px solid var(--border); border-radius: 4px;
      color: var(--text); cursor: pointer; line-height: 1;
    }
    .reload-btn:hover { background: #2a3340; }

    .preflight-list { list-style: none; padding: 0; margin: 0; }
    .preflight-list li { padding: 4px 0; border-bottom: 1px solid #14181f; display: grid;
                         grid-template-columns: 16px 1fr; gap: 6px; }
    .preflight-list li:last-child { border-bottom: none; }
    .preflight-list .ok-mark { color: var(--ok); font-weight: 700; }
    .preflight-list .fail-mark { color: var(--err); font-weight: 700; }
    .preflight-list .detail { color: var(--muted); font-size: 12px; }

    .small { font-size: 12px; color: var(--muted); }
    .grow { flex: 1; min-width: 0; }
    .hidden { display: none; }
  </style>
</head>
<body>
<header>
  <h1>Transporter <span class="sub">control + test dashboard</span></h1>
  <div class="pills">
    <span class="pill" id="mqtt-pill"><span class="dot"></span><span id="mqtt-text">MQTT…</span></span>
    <span class="pill" id="state-pill"><span class="dot"></span><span id="state-text">state: —</span></span>
    <span class="small" id="broker-label">{{ broker_label }}</span>
  </div>
</header>

<main>
  <!-- LEFT COLUMN: controls -->
  <div>
    <div class="card">
      <h2>Emergency</h2>
      <button class="danger stop-big" data-cmd="stop">STOP — halt base now</button>
      <div class="row" style="margin-top:8px;">
        <button class="warn"  data-cmd="pause">Pause &amp; idle</button>
        <button             data-cmd="resume">Resume</button>
        <button             data-cmd="recover">Recover (clear ERROR)</button>
      </div>
    </div>

    <div class="card">
      <h2>Dispatch</h2>
      <div class="row">
        <label>From</label>
        <select id="from-station">
          {% for s in stations %}<option value="{{s}}">{{s}}</option>{% endfor %}
        </select>
        <label>To</label>
        <select id="to-station">
          {% for s in stations %}<option value="{{s}}">{{s}}</option>{% endfor %}
        </select>
        <button class="primary" id="btn-dispatch">Dispatch →</button>
      </div>
      <div class="small" style="margin-top:6px;">
        Sends <code>dispatch</code>. The transporter must be IDLE.
      </div>
    </div>

    <div class="card">
      <h2>Test legs (open-loop drive)</h2>
      <div class="row">
        <label>Leg</label>
        <select id="leg-name">
          {% for c in leg_commands %}<option value="{{c}}">{{c}}</option>{% endfor %}
        </select>
        <label>Distance (m)</label>
        <input type="number" id="leg-distance" value="0.10" step="0.01" min="0" style="width:90px;">
        <button class="primary" id="btn-leg">Run leg</button>
      </div>
      <div class="small" style="margin-top:6px;">
        Direct test of <code>go_to_assembler</code> / <code>go_to_drop_off</code> /
        <code>return_to_base</code>. No vision involved — pure timed strafe.
      </div>
    </div>

    <div class="card">
      <h2>Diagnostics</h2>
      <div class="row">
        <button data-cmd="preflight">Preflight</button>
        <button data-cmd="get_odometry">Get odometry</button>
      </div>
      <div id="preflight-box" class="hidden" style="margin-top:10px;">
        <div class="small" style="margin-bottom:4px;">Last preflight</div>
        <ul class="preflight-list" id="preflight-list"></ul>
      </div>
      <div id="odometry-box" class="hidden" style="margin-top:10px;">
        <div class="small" style="margin-bottom:4px;">Last odometry (body frame)</div>
        <dl class="kv" id="odometry-kv"></dl>
      </div>
    </div>

    <div class="card">
      <h2>Custom command</h2>
      <div class="row">
        <input type="text" id="custom-event" placeholder="event (e.g. dispatch)" class="grow">
      </div>
      <div class="row" style="margin-top:6px;">
        <input type="text" id="custom-params" placeholder='params JSON, e.g. {"distance": 0.2}' class="grow">
        <button id="btn-custom">Send</button>
      </div>
      <div class="small" style="margin-top:6px;">
        For poking at events that aren't on a button. Free-form.
      </div>
    </div>
  </div>

  <!-- RIGHT COLUMN: telemetry + camera -->
  <div>
    {% if stream_enabled %}
    <div class="card">
      <h2>
        Camera preview
        <button id="stream-reload" class="reload-btn" title="Reload stream">↻</button>
      </h2>
      <div class="stream-wrap" id="stream-wrap">
        <img id="stream-img" alt="MJPEG stream"
             onload="onStreamLoad()" onerror="onStreamError()">
        <div id="stream-placeholder" class="placeholder hidden">
          <div>Camera not reachable at <code id="stream-url-text">…</code>.</div>
          <div style="margin-top:8px;">
            Make sure <code>factory_edge.py</code> is running with the MJPEG
            preview enabled (it's on by default at port 8080; pass
            <code>--no-stream</code> to disable).
          </div>
          <div style="margin-top:8px;">
            <a id="stream-link" href="#" target="_blank" rel="noopener">Open in new tab</a>
          </div>
        </div>
      </div>
      <div class="small" id="stream-source" style="margin-top:6px;"></div>
    </div>
    {% endif %}

    <div class="card">
      <h2>Last state</h2>
      <dl class="kv" id="state-kv"><dt>—</dt><dd>—</dd></dl>
    </div>

    <div class="card">
      <h2>Last result</h2>
      <pre id="result-pre">—</pre>
    </div>

    <div class="card">
      <h2>Last error</h2>
      <pre id="error-pre">—</pre>
    </div>

    <div class="card">
      <h2>Live event log</h2>
      <div id="log"></div>
    </div>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);

// Server-injected stream config. Empty stream_url ⇒ derive from this page's
// hostname so opening the dashboard from anywhere that can reach the Pi just
// works — no flag plumbing required.
const STREAM_ENABLED = {{ 'true' if stream_enabled else 'false' }};
const STREAM_URL_OVERRIDE = {{ (stream_url or '') | tojson }};
const STREAM_PORT = {{ stream_port }};

function streamUrl() {
  if (STREAM_URL_OVERRIDE) return STREAM_URL_OVERRIDE;
  return `http://${window.location.hostname}:${STREAM_PORT}/`;
}

function loadStream() {
  if (!STREAM_ENABLED) return;
  const url = streamUrl();
  $('stream-url-text').textContent = url;
  $('stream-link').href = url;
  $('stream-source').textContent = STREAM_URL_OVERRIDE
    ? `source: --stream-url=${url}`
    : `source: auto (port ${STREAM_PORT} on ${window.location.hostname})`;
  // Cache-bust the reconnect so the browser doesn't reuse a dead MJPEG socket.
  $('stream-img').src = url + (url.includes('?') ? '&' : '?') + 't=' + Date.now();
  $('stream-img').classList.remove('hidden');
  $('stream-placeholder').classList.add('hidden');
}

function onStreamLoad() {
  $('stream-img').classList.remove('hidden');
  $('stream-placeholder').classList.add('hidden');
}

function onStreamError() {
  $('stream-img').classList.add('hidden');
  $('stream-placeholder').classList.remove('hidden');
}

if (STREAM_ENABLED) {
  document.addEventListener('DOMContentLoaded', () => {
    loadStream();
    $('stream-reload').addEventListener('click', loadStream);
  });
}

function setPill(el, text, cls) {
  el.querySelector('.dot');
  el.className = 'pill ' + cls;
  el.querySelector('span:last-child').textContent = text;
}

function setMqtt(connected) {
  const pill = $('mqtt-pill');
  if (connected) setPill(pill, 'MQTT connected', 'ok');
  else setPill(pill, 'MQTT disconnected', 'err');
}

function setState(stateName) {
  const pill = $('state-pill');
  if (!stateName) { setPill(pill, 'state: —', ''); return; }
  const cls = stateName === 'IDLE' ? 'idle'
            : stateName === 'DELIVERING' ? 'warn'
            : stateName === 'ERROR' ? 'err' : '';
  setPill(pill, 'state: ' + stateName, cls);
}

function renderState(s) {
  if (!s) return;
  setState(s.state);
  const dl = $('state-kv');
  dl.innerHTML = '';
  const fields = [
    ['machine', s.machine], ['state', s.state],
    ['previous_state', s.previous_state],
    ['event', s.event], ['task_complete', s.task_complete],
    ['timestamp', s.timestamp],
  ];
  for (const [k, v] of fields) {
    if (v === undefined || v === null) continue;
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = String(v);
    dl.appendChild(dt); dl.appendChild(dd);
  }
}

function renderResult(r) {
  $('result-pre').textContent = r ? JSON.stringify(r, null, 2) : '—';
}

function renderError(e) {
  $('error-pre').textContent = e ? JSON.stringify(e, null, 2) : '—';
}

function renderOdometry(o) {
  if (!o) { $('odometry-box').classList.add('hidden'); return; }
  $('odometry-box').classList.remove('hidden');
  const dl = $('odometry-kv'); dl.innerHTML = '';
  for (const [k, v] of Object.entries(o)) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = (typeof v === 'number') ? v.toFixed(4) : String(v);
    dl.appendChild(dt); dl.appendChild(dd);
  }
}

function renderPreflight(p) {
  if (!p || !p.checks) { $('preflight-box').classList.add('hidden'); return; }
  $('preflight-box').classList.remove('hidden');
  const ul = $('preflight-list'); ul.innerHTML = '';
  for (const c of p.checks) {
    const li = document.createElement('li');
    const mark = document.createElement('span');
    mark.className = c.ok ? 'ok-mark' : 'fail-mark';
    mark.textContent = c.ok ? '✓' : '✗';
    const body = document.createElement('div');
    body.innerHTML = '<div><strong>' + c.name + '</strong></div>'
                   + (c.detail ? '<div class="detail">' + c.detail + '</div>' : '');
    li.appendChild(mark); li.appendChild(body); ul.appendChild(li);
  }
}

function appendLog(evt) {
  const div = document.createElement('div'); div.className = 'entry';
  const ts = document.createElement('span'); ts.className = 'ts'; ts.textContent = (evt.ts || '').slice(11, 19);
  const k  = document.createElement('span'); k.className = 'kind ' + evt.kind; k.textContent = evt.kind;
  const b  = document.createElement('span'); b.className = 'body';
  b.textContent = summariseEvent(evt);
  div.appendChild(ts); div.appendChild(k); div.appendChild(b);
  const log = $('log');
  log.appendChild(div);
  // Cap rendered DOM length so the page stays responsive after a long session.
  while (log.children.length > 300) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

function summariseEvent(evt) {
  const d = evt.data || {};
  switch (evt.kind) {
    case 'state':           return (d.previous_state ? d.previous_state + ' → ' : '') + d.state + (d.event ? '  (event=' + d.event + ')' : '');
    case 'error':           return d.error + ': ' + (d.context || '');
    case 'result':          return d.event + ' → ' + (d.ok ? 'OK' : 'FAIL') + (d.detail ? '  ' + d.detail : '');
    case 'command_sent':    return d.event + '  ' + JSON.stringify(d.params);
    case 'mqtt_connected':  return 'connected to ' + d.host + ':' + d.port;
    case 'mqtt_disconnected': return 'broker dropped';
    default:                return JSON.stringify(d);
  }
}

async function sendCommand(event, params) {
  try {
    const r = await fetch('/api/command', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({event, params: params || {}}),
    });
    const j = await r.json();
    if (!j.ok) alert('Command rejected: ' + (j.error || 'unknown'));
  } catch (e) {
    alert('Failed to send command: ' + e);
  }
}

// Wire up all the simple buttons with a data-cmd attribute.
document.querySelectorAll('button[data-cmd]').forEach((b) => {
  b.addEventListener('click', () => sendCommand(b.dataset.cmd, {}));
});

$('btn-dispatch').addEventListener('click', () => {
  const from = $('from-station').value, to = $('to-station').value;
  if (from === to) { alert('From and to stations must differ.'); return; }
  sendCommand('dispatch', {from_station: from, to_station: to});
});

$('btn-leg').addEventListener('click', () => {
  const event = $('leg-name').value;
  const distance = parseFloat($('leg-distance').value);
  if (!isFinite(distance) || distance <= 0) { alert('Distance must be > 0.'); return; }
  sendCommand(event, {distance});
});

$('btn-custom').addEventListener('click', () => {
  const event = $('custom-event').value.trim();
  if (!event) { alert('Need an event name.'); return; }
  let params = {};
  const raw = $('custom-params').value.trim();
  if (raw) {
    try { params = JSON.parse(raw); } catch (e) { alert('Params must be valid JSON.'); return; }
  }
  sendCommand(event, params);
});

async function bootstrap() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();
    setMqtt(s.mqtt_connected);
    if (s.last_state) renderState(s.last_state);
    if (s.last_result) renderResult(s.last_result);
    if (s.last_error)  renderError(s.last_error);
    if (s.last_odometry) renderOdometry(s.last_odometry);
    if (s.last_preflight) renderPreflight(s.last_preflight);
    for (const e of (s.events || []).slice(-50)) appendLog(e);
  } catch (e) {
    console.warn('bootstrap failed', e);
  }
}

function startStream() {
  const es = new EventSource('/api/events');
  es.onmessage = (m) => {
    let evt; try { evt = JSON.parse(m.data); } catch { return; }
    appendLog(evt);
    switch (evt.kind) {
      case 'state':              renderState(evt.data); break;
      case 'error':              renderError(evt.data); break;
      case 'result':
        renderResult(evt.data);
        if (evt.data.event === 'get_odometry' && evt.data.ok) renderOdometry(evt.data.data);
        if (evt.data.event === 'preflight') renderPreflight(evt.data.data);
        break;
      case 'mqtt_connected':     setMqtt(true); break;
      case 'mqtt_disconnected':  setMqtt(false); break;
    }
  };
  es.onerror = () => {
    // EventSource auto-reconnects; just reflect the gap in the UI.
    setMqtt(false);
  };
}

bootstrap().then(startStream);
</script>
</body>
</html>
"""


# ── Entrypoint ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Transporter dashboard")
    parser.add_argument("--broker", default="10.8.210.18", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument(
        "--http-host",
        default="0.0.0.0",
        help="Address to bind the dashboard HTTP server (default: all interfaces)",
    )
    parser.add_argument(
        "--http-port", type=int, default=8000, help="HTTP port for the dashboard"
    )
    parser.add_argument(
        "--stream-url",
        default=None,
        help="Explicit MJPEG stream URL. Default: auto-derive from the browser's "
        "address with --stream-port (so opening the dashboard from the Pi just "
        "works). Pass --no-stream to hide the camera tile entirely.",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=8080,
        help="Port the MJPEG preview is served on by factory_edge (default: 8080). "
        "Used when --stream-url is not given.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Hide the camera tile in the UI.",
    )
    args = parser.parse_args()

    state = DashboardState()
    broadcaster = Broadcaster()
    bridge = MqttBridge(args.broker, args.mqtt_port, state, broadcaster)
    bridge.start()

    app = create_app(
        bridge,
        state,
        broadcaster,
        stream_url=args.stream_url,
        stream_port=args.stream_port,
        stream_enabled=not args.no_stream,
        broker_label=f"{args.broker}:{args.mqtt_port}",
    )

    if args.no_stream:
        stream_descr = "disabled"
    elif args.stream_url:
        stream_descr = f"override={args.stream_url}"
    else:
        stream_descr = f"auto (port {args.stream_port} on browser host)"
    log.info(
        "Dashboard up on http://%s:%d/  (broker=%s:%d, camera=%s)",
        args.http_host, args.http_port,
        args.broker, args.mqtt_port,
        stream_descr,
    )
    try:
        # threaded=True so SSE long-poll connections don't block command POSTs.
        app.run(host=args.http_host, port=args.http_port, threaded=True, use_reloader=False)
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
