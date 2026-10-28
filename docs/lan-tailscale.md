# Host startup: LAN and Tailscale

2026-10-24: user requested both bindings and an executable startup script.

Run from the project directory:

```bash
./start-web.sh
```

The script selects the project virtual environment and starts real generation
on up to three explicit IPv4 addresses, sharing one serial job queue:

- Localhost: http://127.0.0.1:8765, always.
- LAN: the private IPv4 address of the default route (`ip -4 route get`), if
  the machine has one. On the development PC, http://192.168.68.103:8765.
- Tailscale: `tailscale ip -4`, if Tailscale is up. On the development PC,
  http://100.98.64.62:8765.

The script also works when invoked by absolute path from another directory.
Stop it with Ctrl+C. It runs in the foreground; no login/boot autostart or
systemd unit is installed. It uses `output/web` and `cache` beneath the project.
It applies the previously used direct-host memory attestation; set
`QWEN_ATTEST_UNBOUNDED_HOST=0` in a memory-limited environment.

Optional overrides:

```bash
QWEN_WEB_PORT=8766 ./start-web.sh
QWEN_LAN_HOST=192.168.68.104 ./start-web.sh
```

`QWEN_TAILSCALE_HOST` can override the Tailscale IP; set either variable to
`""` to leave that network out. Binding any requested address must succeed,
otherwise startup closes earlier listeners and refuses, which is why only
addresses found on this machine are passed by default. There is
no wildcard fallback. The executable bit is tracked in git.

The underlying server now accepts repeated `--host` options for explicit IPv4
loopback, RFC1918 LAN and Tailscale/CGNAT literals. IPv6, public IPs, hostnames
and wildcard binds remain refused. Host must match the particular listener IP
and actual port; `localhost` is an alias only on the 127.0.0.1 listener. Origin
must match that exact origin, including its port.

Browsers may also use this machine's own names, listed in
`cache/web-settings.json` and accepted on every listener:

```json
{"host_names": ["tiktak-ai", "tiktak-ai.local", "tiktak-ai.tail3e3fc8.ts.net"]}
```

The first `./start-web.sh` writes that file with the hostname,
`<hostname>.local` (mDNS) and the Tailscale MagicDNS name, so
`http://tiktak-ai.local:8765/` works. Edit it and restart to change the list;
`[]` accepts IP addresses only. A file that is not exactly that shape stops the
server at startup. `qwen-web --host-name NAME` adds names for one run. Names
are matched as text and never resolved; any other name is still refused, which
is what stops a DNS-rebinding page from reaching the server under its own
domain. Existing JSON, X-Qwen-UI,
body limits, response headers, image verification and path protections remain.
All listeners use the same worker and cancel/close it once at shutdown.

The app has no login layer. This is the requested exposure to peers able to
reach these LAN/Tailscale addresses. No firewall, Tailscale ACL, router, or
unrelated service configuration was changed.

Verified on this host: interface addresses, three actual port8765 listeners,
generation-enabled status and same-origin planning on each, and rejection of
foreign Host and Origin. This verifies local connections to those addresses;
an independent remote-client test was not performed. Bind-policy, partial-startup
cleanup and shared-worker/shutdown regressions are in `tests/test_web_bind.py`.
Real Chrome also recognized the generation-enabled workspace through both LAN
and Tailscale IPs on this host. No additional GPU generation was needed for the
bind change; the previously verified W3 engine path is unchanged.

Restart testing also fixed an existing server-construction issue: worker state
is initialized before the socket constructor, so a failed bind reports its
original error and closes cleanly. SO_REUSEADDR now permits clean restarts while
old connections are in TIME_WAIT; SO_REUSEPORT is not enabled.

Final validation: guarded CPU suite **2567 passed, 2 skipped** (23.17 s),
HTTP/real-Chrome planning regression **105/105 passed**. A separate live-socket
check confirmed a second server gets a clean bind error while the first runs,
then succeeds immediately on the same port after the first stops, even after
serving a request.
