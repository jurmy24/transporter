#!/usr/bin/env bash
# Launch the transporter factory edge.
#
# Usage:
#   ./run.sh                       # uses defaults below
#   ./run.sh --broker 10.8.30.43   # override any flag on the CLI
#
# Environment overrides:
#   BROKER       MQTT broker host (default: 10.8.30.43)
#   MQTT_PORT    MQTT broker port (default: 1883)
#   SERIAL_PORT  Serial port for the motor bus (default: /dev/ttyACM0)
#   ROBOT_ID     Robot ID (default: transporter)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BROKER="${BROKER:-10.8.30.43}"
MQTT_PORT="${MQTT_PORT:-1883}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-transporter}"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

exec python factory_edge.py \
    --broker "$BROKER" \
    --mqtt-port "$MQTT_PORT" \
    --port "$SERIAL_PORT" \
    --id "$ROBOT_ID" \
    "$@"
