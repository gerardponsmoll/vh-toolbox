"""Open a demo in the browser, starting its server if it is not running yet.

    python demo.py                 # list the demos
    python demo.py camera          # open the camera demo (by name or number)
    python demo.py camera --fresh  # restart it first, back to its initial state
    python demo.py --stop          # stop all demo servers

A demo starts in under a second. It keeps running on its own fixed port after
this command returns, so opening it again shows it as you left it (--fresh
starts over).
"""

from __future__ import annotations

import argparse
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = Path(tempfile.gettempdir()) / "vh-demos"

# name: (lecture, script, port), in the order they come up in the lectures.
DEMOS = {
    "smpl": ("1", "01_smpl_viewer.py", 8080),
    "facade": ("2.1", "02_facade.py", 8081),
    "camera": ("2.1", "02_camera.py", 8082),
    "lens": ("2.1", "02_lens.py", 8083),
    "telephoto": ("2.1", "02_telephoto.py", 8084),
    "rotations": ("2.2", "02_rotations.py", 8085),
    "chain": ("2.2", "02_kinematic_chain.py", 8086),
    "ik": ("2.2", "02_smpl_ik.py", 8087),
}


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def listener(port: int) -> tuple[str, str] | None:
    """The process listening on a port: (pid, command), if lsof can tell."""
    if not shutil.which("lsof"):
        return None
    pids = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    if not pids:
        return None
    command = subprocess.run(["ps", "-o", "command=", "-p", pids[0]], capture_output=True, text=True).stdout.strip()
    return pids[0], command


def stop(name: str) -> None:
    _, script, port = DEMOS[name]
    owner = listener(port)
    if owner and script in owner[1]:
        subprocess.run(["kill", owner[0]])
        while port_in_use(port):
            time.sleep(0.1)


def start(name: str) -> None:
    """Start the server if needed and wait until it answers."""
    _, script, port = DEMOS[name]
    if port_in_use(port):
        owner = listener(port)
        if owner is None or script in owner[1]:
            return
        stop_other = "demos/" in owner[1]  # an old demo started on the wrong port
        if not stop_other:
            raise SystemExit(f"Port {port} is used by another program: {owner[1]}")
        subprocess.run(["kill", owner[0]])
        while port_in_use(port):
            time.sleep(0.1)
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"{name}.log"
    process = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "demos" / script), "--port", str(port)],
        cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    # The demos print "Open http://..." once their scene is built (the port opens earlier).
    while "Open http" not in log.read_text():
        if process.poll() is not None:
            raise SystemExit(f"{script} stopped while starting:\n{log.read_text()[-2000:]}")
        time.sleep(0.1)


def tabs_open(name: str) -> int:
    """How many browser tabs are connected to the running demo, from viser's log lines."""
    log = LOGS / f"{name}.log"
    counts = re.findall(r"Connection (?:opened|closed) \(\d+, (\d+) total\)", log.read_text()) if log.exists() else []
    return int(counts[-1]) if counts else 0


def resolve(key: str) -> str:
    names = list(DEMOS)
    if key.isdigit() and 1 <= int(key) <= len(names):
        return names[int(key) - 1]
    matches = [n for n in names if n.startswith(key.lower()) or key.lower() in DEMOS[n][1]]
    if len(matches) != 1:
        raise SystemExit(f"Unknown demo {key!r}. Demos: {', '.join(names)}")
    return matches[0]


def print_list() -> None:
    for k, (name, (lecture, script, port)) in enumerate(DEMOS.items(), start=1):
        state = "running" if port_in_use(port) else ""
        print(f"{k}  {name:10s} lecture {lecture:4s} demos/{script:24s} http://localhost:{port}  {state}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("demo", nargs="?", help="name or number, see the list")
    parser.add_argument("--fresh", action="store_true", help="restart the demo first")
    parser.add_argument("--stop", action="store_true", help="stop all demo servers")
    args = parser.parse_args()

    if args.stop:
        for name in DEMOS:
            stop(name)
    elif args.demo:
        name = resolve(args.demo)
        tabs = tabs_open(name) if port_in_use(DEMOS[name][2]) else 0
        if args.fresh:
            stop(name)
        start(name)
        if args.fresh and tabs:
            return  # the open tab reconnects to the restarted demo by itself
        webbrowser.open(f"http://localhost:{DEMOS[name][2]}")
        return
    print_list()


if __name__ == "__main__":
    main()
