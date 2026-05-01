# Transporter

Three-omniwheel mobile base, controlled over MQTT.

## Run the edge (on the Pi)

```
./run.sh
```

The MJPEG camera preview is **on by default** at port 8080 (open
`http://<pi>:8080/` to view it directly). Pass `--no-stream` to disable.

Underneath, this is just:

```
python factory_edge.py --broker <broker> --port /dev/ttyACM0
```

## Run the dashboard (anywhere with broker access)

A web UI for sending commands and watching state.

```
./dashboard.sh
```

Then open `http://<host>:8000/`. The camera tile **auto-discovers** the
MJPEG stream by looking at port 8080 on the same host you opened the
dashboard from, so the typical flow — start `factory_edge.py` and
`dashboard.sh` on the Pi, then open `http://<pi>:8000/` from any browser —
just works.

Override only if your setup is unusual:

```
./dashboard.sh --stream-url http://10.8.210.18:8080/
./dashboard.sh --stream-port 9000      # if you changed factory_edge's port
./dashboard.sh --no-stream             # hide the tile entirely
```

Features:

- Live state / error / result updates via Server-Sent Events.
- Quick actions: STOP, pause, resume, recover.
- Dispatch between stations (`picker`, `manipulator`, `delivery`).
- Test legs (`go_to_assembler`, `go_to_drop_off`, `return_to_base`)
  with adjustable distance.
- Diagnostics: preflight + odometry readouts.
- Free-form custom command box for anything not on a button.
- Embedded MJPEG camera preview with reload button.

`uv sync` once before either script to install deps.
