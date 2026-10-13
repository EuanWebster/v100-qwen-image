# Pure generation memory-policy groundwork

`qwen_engine.memory` adds typed immutable exact-match evidence and refusal
arithmetic. It does **not** probe hardware, collect/persist calibration,
load components, change the existing encoder guard, wire the CLI/adapter,
or prove fit. No expert override, interpolation or automatic quality/mode
change. Real autodiscovery still cannot prove complete cgroup scope and
refuses all real encoder loads in this deployment.

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
uninitialized-CUDA/unknown-workspace state. Actual CUDA identity acceptance,
resource/peak collectors and default-loader integration remain unimplemented.

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
Host footprint must include the **owned process tree**, not just parent
RSS while an encoder child runs. Summed child RSS may conservatively double
count shared pages. A future collector must actually measure these interval
peaks with adequate coverage; a one-off counter is not a peak measurement.
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

113 guarded pure tests cover every descriptor/device/cache drift, unresolved
backend/dtypes/steps, staleness/future/clock faults, conditioning unknowns,
incomplete/old records, parent-only/Torch-only peaks, finite/zero/unknown
cgroup constraints, exact2K refusal, bounds, worst-case/headroom arithmetic
and typed invalid inputs. Positive arithmetic cases use **fabricated trusted
fixtures**, not measured hardware/model data. No calibration persistence or
existing benchmark schema changed. Collector, probe, reviewed adapter and
CLI/job integration remain separate milestones.
