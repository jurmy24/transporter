#!/usr/bin/env bash
# Launch the transporter factory edge.
#
# Usage:
#   ./run.sh                       # uses defaults below
#   ./run.sh --broker 10.8.210.18   # override any flag on the CLI
#
# Environment overrides:
#   BROKER       MQTT broker host (default: 10.8.210.18)
#   MQTT_PORT    MQTT broker port (default: 1883)
#   SERIAL_PORT  Serial port for the motor bus (default: /dev/ttyACM0)
#   ROBOT_ID     Robot ID (default: transporter)
#
# Dependencies: run `uv sync` once (creates .venv from pyproject.toml).
# If `uv` is on PATH, this script uses `uv run`; otherwise it activates .venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="${BROKER:-10.8.210.18}"
MQTT_PORT="${MQTT_PORT:-1883}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-transporter}"

if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
elif [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
    PYTHON=(python)
else
    PYTHON=(python)
fi

exec "${PYTHON[@]}" factory_edge.py \
    --broker "$BROKER" \
    --mqtt-port "$MQTT_PORT" \
    --port "$SERIAL_PORT" \
    --id "$ROBOT_ID" \
    "$@"
