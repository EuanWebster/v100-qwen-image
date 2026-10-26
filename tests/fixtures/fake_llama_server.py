"""A stand-in for ``llama-server`` for the guarded suite's process tests.

It does what the real server does that ``pe_llamacpp`` depends on: it stays
alive, it writes a banner naming the argv it was given, and it dies on the
signal its parent sends to the process group. It serves NO HTTP: the CPU guard
(``tests/cpu_guard/sitecustomize.py``) denies loopback TCP socket construction
in the pytest process and in every child it spawns, so a fake that bound a
port could not run here — the HTTP half is exercised over an AF_UNIX
socketpair inside ``tests/test_pe_llamacpp.py``, against the real
``http.client`` code path.

``FAKE_SERVER_SECONDS`` is how long it stays alive (30 by default: the tests
kill it well before that, and a leak would be visible as a stuck child).
"""
from __future__ import annotations

import os
import sys
import time

print("fake llama-server argv: " + " ".join(sys.argv[1:]), flush=True)
print("fake llama-server: awaiting the health poll", flush=True)
time.sleep(float(os.environ.get("FAKE_SERVER_SECONDS", "30")))
