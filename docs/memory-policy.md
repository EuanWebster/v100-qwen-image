# Pure generation memory-policy groundwork

`qwen_engine.memory` adds typed immutable exact-match evidence and refusal
arithmetic. The policy module itself does **not** probe hardware, load
components, change the existing encoder guard, or prove fit; the separate
opt-in collector (`qwen_engine.calibration`, enabled with
`--calibration-record`) measures and persists records, and every record still
has to pass these exact-match and age gates. No expert override,
interpolation or automatic quality/mode change. Real autodiscovery still cannot prove complete cgroup scope and
refuses all real encoder loads in this deployment unless the operator
attestation below is used.

## The opt-in operator host-scope attestation

`--attest-unbounded-host` (CLI) / `attested_unbounded_host=True` (worker and
encoder client) / `trust_unbounded_host=True` (loader adapter constructor) is
the single deliberate escape from the scope gate. It is an **operator
statement about scope**, never a measurement and never a number:

- The boolean declares that the encoder process tree is not confined by a
  cgroup memory limit. It cannot be derived from the environment, config
  convenience or a caller-supplied snapshot.
- The availability figure is read *inside* the gate from `/proc/meminfo`
  (`MemAvailable`) at decision time, so an attestation cannot smuggle in a
  fabricated budget; an unreadable value still refuses.
- It is off by default and mutually exclusive with an injected
  `MemorySnapshot`. The preflight report carries
  `ATTESTED_UNBOUNDED_REASON`, the CLI writes a `host_scope_attested` event
  and prints a notice, and the worker's two memory gates both receive it.
- A hidden cap would not be detected. The attestation is exactly as
  trustworthy as the operator making it; it is appropriate for a
  single-user workstation where the operator knows the machine, not for a
  shared or containerized deployment. Nothing else in this document is
  relaxed by it: headroom, GPU evidence, calibration and quality acceptance
  remain separate gates.

## Identity and completeness

Every descriptor includes phase; exact denoise step index; resolution/token
grid; total steps; measured positive/active-negative prompt lengths; CFG
activity (not an assumed doubled batch); resolved attention backend; exact
quant-format label; residency; VAE tiling; encoder/execution dtypes; KV-cache
policy; embedding-cache hit/miss (cold miss by default); model revision; complete asset-set digest; runtime/implementation
identity digest. Missing token lengths do not become zero or inferred text
length. AUTO is not a resolved backend. Current execution profile is T2I,
batch1, FP16 Volta; other profiles need a separate reviewed implementation.
The future runtime-identity producer must include platform/CPU/host execution
identity, precision/allocator/backend settings and relevant thread/environment
configuration as well as package/source identity. The encoder's current
package-metadata digest alone is **not** this broader generation identity.
A warm embedding-cache observation cannot authorize a cold encoder miss;
the actual validated branch must match, otherwise use conservative cold
evidence or refuse. The current-process runtime fingerprint now has a CPU
witness ([runtime-identity.md](runtime-identity.md)), including explicit
uninitialized-CUDA/unknown-workspace state. The opt-in collector now binds
the CUDA/NVML device identity, samples device-wide and owned-tree peaks and
reads the default loader's measured conditioning facts
([calibration-collector-2026-10-24.md](calibration-collector-2026-10-24.md));
job/web integration remains separate.

Device identity includes UUID, architecture, driver and CUDA build. The
resource provider must independently verify actual CUDA kernel execution
for that exact runtime/device, not use wheel architecture flags. Current
policy accepts only sm_70/CUDA12.6 evidence, not a newly discovered backend.
Identity/counters supplied by a caller are **trusted measurement claims**,
not unforgeable tokens or proof that installed bytes/driver/hardware are
authentic. Future adapter/probe code must establish that trust boundary.

Fresh resource counters must include a trusted complete MemorySnapshot
budget and GPU total/free memory. Unknown scope/host/free counters refuse;
advisory namespace numbers and total host RAM cannot grant a budget. The
provider must use the oldest component-read time for captured_at, and rerun
the query just before each phase/step. Explicit age limits and explicit
host/GPU headroom are required; no guessed default or fake clock on failure.

Eligible calibration requires an exact descriptor/device match and fresh,
not future, observation time. A matching stale/incomplete observation blocks,
even alongside a good sample; a collector must explicitly select a reviewed
fresh set. Bench v1 records are not silently upgraded: they lack required
runtime/device and footprint evidence and are rejected as input.

GPU footprint must include **device-wide used-memory peaks** (context and
non-Torch allocations), not merely Torch allocator reserved/allocated peaks.
host footprint must include the **owned process tree**, not just parent
RSS while an encoder child runs. Summed child RSS may conservatively double
count shared pages. The opt-in collector measures these interval peaks per
phase (50 ms NVML by device UUID; owned-tree RSS); coverage is
interval-limited — a spike shorter than the interval can be missed — so the
record is exact evidence, not a guarantee of continuous observation.
Missing measurements remain unknown. Reserved/allocated/tree consistency
is validated; current main-process RSS cannot be zero.

## Arithmetic and remaining implementation

Use the worst complete matching peak for each memory domain plus configured
headroom, and compare its full absolute footprint to current availability.
Never average downward, interpolate across resolutions/steps/prompt lengths,
subtract an invented resident baseline or use stale metadata as fit proof.
This intentionally may refuse unnecessarily. Even an arithmetic pass is only
a conservative prerequisite, not a guarantee of future allocation success,
fragmentation freedom, numerical quality or resource reservation against
other services. Caller code must still enforce all hardware/assets/fidelity
gates. Unknown/unmeasured fields never enable a pass.

114 guarded policy tests cover every descriptor/device/cache drift, unresolved
backend/dtypes/steps, staleness/future/clock faults, conditioning unknowns,
incomplete/old records, parent-only/Torch-only peaks, finite/zero/unknown
cgroup constraints, exact2K refusal, bounds, worst-case/headroom arithmetic
and typed invalid inputs. Positive arithmetic cases here use **fabricated
trusted fixtures**; the measured evidence lives in
[calibration-collector-2026-10-24.md](calibration-collector-2026-10-24.md).
The collector, exactness bridge and default-loader/CLI integration have
landed with 39 calibration tests; job/web integration remains a separate
milestone. Legacy bench records stay loadable but ineligible.
