# Transporter

Keyboard teleoperation for LeKiwi's omniwheel base (wheels only, no arm).

`transporter.py` is a wheels-only fork of lerobot's `LeKiwi` -- it only registers the 3 base motors (IDs 7, 8, 9) so it works without an arm connected.

## Usage

**On the Pi** (SSH in first):

```bash
python host.py
```

**On your laptop:**

```bash
python teleoperate.py --ip <pi-ip-address>
```

Use WASD to drive, Z/X to rotate, R/F to change speed. `Ctrl+C` to stop.

## Options

```bash
# host.py
python host.py --port /dev/ttyACM0 --id biden_kiwi --connection-time 600

# teleoperate.py
python teleoperate.py --ip 172.20.10.2 --id biden_kiwi
```

## Setup

```bash
uv sync
```
