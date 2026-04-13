# Transporter

Keyboard teleoperation for LeKiwi's omniwheel base. Two scripts, one runs on the Pi, one on your laptop.

## Usage

**On the Pi** (SSH in first):

```bash
python host.py
```

**On your laptop:**

```bash
python teleoperate.py --ip <pi-ip-address>
```

Use arrow keys / WASD to drive. `Ctrl+C` to stop.

## Options

```bash
# host.py
python host.py --robot.port=/dev/ttyACM0 --robot.id=biden_kiwi

# teleoperate.py
python teleoperate.py --ip 172.20.10.2 --id biden_kiwi
```

## Setup

```bash
uv sync
```

Only dependency is `lerobot[feetech]`.
