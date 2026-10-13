# CPU local-only worker / loader milestone

Codex, 2026-10-13; no new Qwen drafting jobs or agents. New encoder_worker.py
and qwen-encoder-worker entrypoint connect memory/runtime/asset identity,
strict bounded no-follow JSON inputs, offline child guard, allowlisted verified
index paths, fixed local-only CPU model/processor loading, upstream helper and
bounded wire response/atomic no-clobber output. Generic encoder transport/cache,
ML locks, actual model assets, services and user's sandbox.yaml untouched.

58 new guarded tests. Both full suites1160 passed+2skip on Python3.11.15/3.13.5.
Pinned-ML worker checks12/12 each, including actual fixed-class from_pretrained
on generated miniature checkpoints in fp16/bf16/fp32, bitexact helper comparison,
CPU placement, nested cache disabled, eval/norm hook cleanup. Processor is still
injected, budget/vision policy privately patched only in disposable test process;
real production policy refusal restored and verified. No downloaded real
checkpoint/processor, GPU, fit, parity or generation evidence. Generic parent
real memory-refusal/reaping/stderr/cleanup path passed. Console help and locked
offline entrypoint setup passed in both lightweight environments.

Real loader caught and fixed use_cache kwarg forwarding into Qwen constructor;
disable nested text config instead. low_cpu_mem_usage is ignored by the pinned
Transformers; default meta init is not a memory fit claim. Child CPU/BLAS/torch
intra/inter-op threads constrained to one (no parent/service settings changed).
Contracts and persistent reproducible checks: docs/encoder-worker.md and
tools/synthetic_worker_check.py. Publication relies on Linux proc-fd paths,
same-filesystem hard links and directory fsync; no overwrite fallback.

Next: complete CPU code/design review and corrections BEFORE activation. Audit
transitive-runtime/cache identity completeness first (current metadata checks
named direct packages, not every locked transitive; tokenizer/Jinja/NumPy drift
must not silently reuse a production cache), parent/child interpreter handoff,
response/input race handling, earlier modules and web scope. No production
enablement or fake dependency/memory bypass: missing/unlocked torchvision,
17,534,339,488B checkpoint >16GiB visible cap and unproven hierarchy completeness
still refuse actual loads. V100 last user-confirmed absent. GPU-specific steps
stay pending real hardware; web remains planning/config-export only.
