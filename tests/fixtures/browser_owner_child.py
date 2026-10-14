"""PRIVATE fork-tree fixture for the browser subreaper; never selects Chrome."""
import os
from pathlib import Path
import signal
import sys
import time

mode, directory = sys.argv[1], Path(sys.argv[2])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = os.fork()
if child == 0:
    grandchild = os.fork()
    label = "child" if grandchild else "grandchild"
    (directory / (label + ".pid")).write_text(str(os.getpid()))
    while True:
        time.sleep(.02)
else:
    (directory / "leader.pid").write_text(str(os.getpid()))
    deadline = time.monotonic() + 5
    while not (directory / "grandchild.pid").exists() or not (directory / "child.pid").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("private fork fixture did not settle")
        time.sleep(.01)
    if mode == "orphan_exit":
        os._exit(0)
    while True:
        time.sleep(.02)
