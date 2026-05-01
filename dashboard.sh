#!/usr/bin/env bash
# Launch the transporter web dashboard.
#
# Usage:
#   ./dashboard.sh                              # uses defaults below
#   ./dashboard.sh --broker 10.8.210.18         # different broker
#
# The camera tile auto-derives its URL from the browser's address with
# port 8080 (matching factory_edge.py's default), so it just works when
# you open the dashboard from the Pi or from a host that can reach it.
# Override with --stream-url, change the port with --stream-port, or
# hide the tile with --no-stream.
#
# Environment overrides:
#   BROKER       MQTT broker host (default: 10.8.210.18)
#   MQTT_PORT    MQTT broker port (default: 1883)
#   HTTP_HOST    Address to bind the HTTP server (default: 0.0.0.0)
#   HTTP_PORT    HTTP port for the dashboard (default: 8000)
#   STREAM_URL   Explicit MJPEG preview URL (default: auto)
#   STREAM_PORT  Camera port used when auto-deriving (default: 8080)
#
# Dependencies: run `uv sync` once.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="${BROKER:-10.8.210.18}"
MQTT_PORT="${MQTT_PORT:-1883}"
HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8000}"
STREAM_URL="${STREAM_URL:-}"
STREAM_PORT="${STREAM_PORT:-8080}"

if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
elif [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
    PYTHON=(python)
else
    PYTHON=(python)
fi

ARGS=(
    --broker "$BROKER"
    --mqtt-port "$MQTT_PORT"
    --http-host "$HTTP_HOST"
    --http-port "$HTTP_PORT"
    --stream-port "$STREAM_PORT"
)
if [[ -n "$STREAM_URL" ]]; then
    ARGS+=(--stream-url "$STREAM_URL")
fi

exec "${PYTHON[@]}" dashboard.py "${ARGS[@]}" "$@"
