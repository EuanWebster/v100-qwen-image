"""Host-memory preflight for the forthcoming CPU encoder child (M2.2b).

Stdlib-only. **This module imports no ML package and reads no model file**:
it answers, before any ML import, weight read or loader call, the single
question "can the numbers presented possibly fit the applicable memory
budget?" — and refuses otherwise.

What it reads (documented Linux cgroup v2 target only — deliberately NOT a
general resource framework):

* **Host available memory**: ``MemAvailable`` from ``/proc/meminfo``,
  accepted only in its documented ``kB`` unit; missing, unparseable,
  sign-prefixed, wrong-unit or beyond-the-interpreter-integer-conversion
  values are **unknown**, never guessed, approximated or multiplied into
  a trusted budget.
* **This process's cgroup v2 memory constraints**: the single ``0::`` line
  of ``/proc/self/cgroup`` locates the cgroup (malformed or ambiguous
  paths — relative, containing ``.``/``..``, duplicate ``0::`` records —
  are refused, never path-joined). Every **visible ancestor** from the
  leaf up to and including the mounted root is walked — a descendant
  cannot escape an ancestor's cap, so the tightest **minimum of
  ``limit - current`` over every finite ``memory.max``** (each level's own
  ``memory.current`` includes sibling usage, so a parent's usage tightens
  the child; an explicit ``max`` at any level means *that level* is
  unlimited and does not end the walk) is computed with truthful per-level
  source notes — but only as an **advisory availability observation**: a
  visible upper bound on the true availability (any hidden ancestor can
  only tighten it, never loosen it). The visible walk can never prove it
  saw the whole hierarchy, so real autodiscovery is fail-closed
  **unknown** (scope gate below) and these numbers are diagnostic refusal
  evidence, never an authoritative complete budget.

Scope gate (fail closed, documented honestly):

* A **missing** ``memory.max`` at ANY visible level is unknown: ``ENOENT``
  does not prove the memory controller is disabled (the directory may be
  missing or mismapped and no controller evidence is read), so missing
  files refuse at every level — never silently skipped, never asserted
  disabled.
* A **present** ``memory.max`` at the visible root — finite or ``max`` —
  does not prove completeness either: a cgroup namespace's ``/`` can be a
  non-root cgroup under a tighter hidden ancestor, and a disclaimer does
  not make allowing a real loader safe. Because root completeness is
  unproven, the real read path returns **unknown / refuse regardless of
  the visible root file's contents**. The default deployment may
  therefore legitimately refuse every real load until trusted complete
  host-budget evidence exists — an honest gate, not a bug to bypass.
* An **explicitly injected** ``MemorySnapshot`` in a finite or unlimited
  state is **trusted caller evidence** for pure budget arithmetic (and is
  how tests exercise pass decisions); nothing here invents a
  root-is-complete default or a production bypass.

Budget rule: the **tighter applicable budget** — with finite caps the
cgroup availability always bounds the decision even when host RAM looks
plentiful (a cap is never substituted away by a larger host number), and
host ``MemAvailable`` is a hard input: **unknown host pressure always
refuses**, even under a finite cap. Anything that cannot rule out a
tighter constraint — no ``0::`` line, unreadable/unparseable/negative
counters, a finite cap without its usage counter (absent current is
**unknown, never zero**), missing/malformed ``MemAvailable``, or the
unproven root scope above — yields a refusal, never a pass. A valid
**zero-byte cap is a zero budget** (a refusal of any positive request),
not a raw failure.

Honest limits — read before trusting a ``proceed``: this is a
**conservative prerequisite, not measured model-fit evidence**. Passing it
alone never proves a load fits: allocator behaviour, conversion transients,
page cache pressure, fragmentation and actual peak residency are not
modelled here; only real measurement under the real loader can demonstrate
fit. The cgroup claim from a real read is likewise **scoped to the visible
hierarchy and never claimed complete**: hidden ancestors are never claimed
proven unconstrained and an absent or present ``memory.max`` is never
mislabeled as proven unlimited or whole-host/root support. The motivating
measurement (2026-10-09): the pinned text-encoder checkpoint totals
17,534,339,488 B while this sandbox's cgroup ``memory.max`` is
17,179,869,184 B (16 GiB) — before expansion or runtime allocations the
weights' allocation lower bound already exceeds even the tightest visible
availability upper bound, and this preflight must say so before anything
expensive or heavy is attempted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from qwen_engine.errors import MemoryGuardError

#: Schema tag of the report document (for events/JSON later).
MEM_PREFLIGHT_SCHEMA = "qwen_engine.mem_preflight.v1"

#: Bytes per scalar for the dtype names this preflight models (lowercase).
#: Anything else is refused as an unknown dtype rather than guessed.
DTYPE_BYTES = {"f32": 4, "fp32": 4, "float32": 4,
               "bf16": 2, "bfloat16": 2,
               "f16": 2, "fp16": 2, "float16": 2}

#: cgroup constraint states.
CGROUP_FINITE = "finite"
CGROUP_UNLIMITED = "unlimited"
CGROUP_UNKNOWN = "unknown"

#: Refusal reason whenever host MemAvailable is unknown — host pressure
#: cannot be ruled out, so no decision proceeds on a cgroup cap alone.
_HOST_UNKNOWN_REASON = (
    "host MemAvailable is unknown: host pressure cannot be ruled out, so "
    "safety cannot be asserted (even alongside a finite cgroup cap)")

#: Scope disclaimer reported with a trusted caller-supplied UNLIMITED
#: cgroup state. Such a state only ever arrives via an explicit snapshot
#: injection — real autodiscovery is fail-closed unknown under the scope
#: gate below and never reports unlimited — so this wording makes no
#: discovery claim ("no finite cap was found"): no hierarchy was walked,
#: the unlimited evidence is the caller's, not independently verified by
#: this module.
_TRUSTED_UNLIMITED_REASON = (
    "unlimited cgroup state from an explicitly injected trusted snapshot: "
    "trusted caller-supplied complete evidence for pure budget arithmetic, "
    "NOT independently verified by this module (no autodiscovery ran on "
    "this path; real discovery stays unknown/refuse under the visible-"
    "root completeness gate, so whole-host/root absence of constraints "
    "is not proven here either way)")

#: Scope gate: a completed visible walk still cannot prove the visible
#: root is the TRUE hierarchy root (a namespace '/' can be a non-root
#: cgroup under a tighter hidden ancestor and no controller/mount
#: evidence is read), so real autodiscovery refuses regardless of what
#: the visible root's memory.max contains. Trusted snapshot injection is
#: the only route to a finite/unlimited state.
_SCOPE_GATE_REASON = (
    "visible cgroup root completeness cannot be proven: a cgroup "
    "namespace '/' can itself be a non-root cgroup under a tighter hidden "
    "ancestor and no controller evidence was read, so the real read path "
    "refuses fail-closed; visible numbers are advisory diagnostics — an "
    "UPPER bound on true availability (hidden ancestors can only tighten "
    "it) — never an authoritative complete budget")


class MemoryPreflightError(MemoryGuardError):
    """The requested load is not provably within the applicable memory
    budget (or the constrained state is unknown). Typed refusal in the
    memory-refusal (exit 3) family; raised BEFORE any ML import, weight
    read or loader call."""


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class MemorySnapshot:
    """One read (or injection) of the machine's memory state. ``None``
    means **unknown** — never assumed plentiful and never substituted by
    zero. ``cgroup_state`` is one of ``"finite"``/``"unlimited"``/
    ``"unknown"``. For a finite state ``cgroup_limit_bytes`` /
    ``cgroup_current_bytes`` are the **chosen** (tightest) level's
    counters from the ancestor walk — an absent usage counter is unknown
    usage against a finite cap, so injection requires it explicitly; a
    valid zero cap is a zero budget, not an error. Validation of an
    injected snapshot is as strict as reading real state. An explicitly
    injected finite/unlimited snapshot states **trusted complete caller
    evidence** for pure budget arithmetic; real autodiscovery instead
    reports the fail-closed ``unknown`` scope gate (module docstring).
    ``advisory_visible_available_bytes`` carries the real read path's
    tightest visible ``limit - current`` as explicitly **advisory
    diagnostic** information — an UPPER bound on true availability (a
    hidden tighter ancestor can only lower it): it never participates in
    ``budget()`` and never enables a pass; trusted injections normally
    leave it ``None``."""

    host_available_bytes: int | None
    cgroup_state: str
    cgroup_limit_bytes: int | None = None
    cgroup_current_bytes: int | None = None
    notes: tuple[str, ...] = ()
    advisory_visible_available_bytes: int | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.notes, (tuple, list))
                or any(not isinstance(note, str) for note in self.notes)):
            raise MemoryPreflightError("notes must be a list/tuple of diagnostic strings")
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.cgroup_state not in (CGROUP_FINITE, CGROUP_UNLIMITED,
                                     CGROUP_UNKNOWN):
            raise MemoryPreflightError(
                f"cgroup_state: must be one of finite/unlimited/unknown, "
                f"got {self.cgroup_state!r}")
        if self.host_available_bytes is not None and not _nonneg_int(
                self.host_available_bytes):
            raise MemoryPreflightError(
                "host_available_bytes: must be None (unknown) or a "
                f"non-negative integer, got {self.host_available_bytes!r}")
        if self.cgroup_state == CGROUP_FINITE:
            if not _nonneg_int(self.cgroup_limit_bytes):
                raise MemoryPreflightError(
                    "a finite cgroup cap requires a non-negative integer "
                    "cgroup_limit_bytes (a zero cap is a zero budget, not "
                    f"an error), got {self.cgroup_limit_bytes!r}")
            if not _nonneg_int(self.cgroup_current_bytes):
                raise MemoryPreflightError(
                    "a finite cgroup cap requires its usage counter: "
                    "cgroup_current_bytes must be a non-negative integer "
                    "— an absent counter is unknown usage, never assumed "
                    f"zero — got {self.cgroup_current_bytes!r}")
        elif self.cgroup_state in (CGROUP_UNLIMITED, CGROUP_UNKNOWN) and (
                self.cgroup_limit_bytes is not None
                or self.cgroup_current_bytes is not None):
            raise MemoryPreflightError(
                "an unlimited/unknown cgroup state carries no counters")
        if self.advisory_visible_available_bytes is not None and (
                not _nonneg_int(self.advisory_visible_available_bytes)):
            raise MemoryPreflightError(
                "advisory_visible_available_bytes: must be None (no "
                "visible finite cap observed) or a non-negative integer, "
                f"got {self.advisory_visible_available_bytes!r}")

    def budget(self) -> tuple[int | None, tuple[str, ...]]:
        """The tighter applicable budget, or ``None`` when no budget can be
        proven; plus human-readable reasons (refusals first)."""
        reasons: list[str] = list(self.notes)
        if self.cgroup_state == CGROUP_UNKNOWN:
            reasons.append(
                "cgroup memory constraint state unknown: a finite cap "
                "cannot be ruled out, so safety cannot be asserted")
            if self.host_available_bytes is None:
                reasons.append(_HOST_UNKNOWN_REASON)
            return None, tuple(reasons)
        if self.host_available_bytes is None:
            return None, tuple(reasons) + (_HOST_UNKNOWN_REASON,)
        if self.cgroup_state == CGROUP_FINITE:
            assert self.cgroup_limit_bytes is not None  # __post_init__
            assert self.cgroup_current_bytes is not None  # __post_init__
            cgroup_available = max(0, self.cgroup_limit_bytes
                                   - self.cgroup_current_bytes)
            if self.host_available_bytes > cgroup_available:
                reasons.append(
                    f"finite cgroup availability {cgroup_available} B is "
                    f"tighter than host MemAvailable "
                    f"{self.host_available_bytes} B: the cap is never "
                    f"substituted away by the host number")
            return min(self.host_available_bytes, cgroup_available), \
                tuple(reasons)
        # Trusted caller-supplied UNLIMITED evidence (injected snapshot —
        # real autodiscovery never reaches this branch): only the host
        # number can decide, with the honest trusted-evidence note.
        if _TRUSTED_UNLIMITED_REASON not in reasons:
            reasons.append(_TRUSTED_UNLIMITED_REASON)
        return self.host_available_bytes, tuple(reasons)


# ---------------------------------------------------------------------------
# Reading real state (paths injectable for tests)
# ---------------------------------------------------------------------------


def read_host_available(meminfo_path: str | Path = "/proc/meminfo"
                        ) -> tuple[int | None, tuple[str, ...]]:
    """``MemAvailable`` in bytes, or unknown. Only the documented
    ``'<int> kB'`` form is trusted; missing, unparseable, sign-prefixed,
    wrong-unit or conversion-limit-exceeding lines are unknown, never
    coerced, approximated or multiplied into a trusted budget."""
    try:
        text = Path(meminfo_path).read_text(encoding="utf-8",
                                            errors="replace")
    except (OSError, ValueError) as exc:
        return None, (f"meminfo unreadable: {exc}",)
    lines = [line for line in text.splitlines() if line.startswith("MemAvailable:")]
    if len(lines) > 1:
        return None, ("meminfo has ambiguous duplicate MemAvailable lines",)
    if lines:
        line = lines[0]
        parts = line.split()
        if (len(parts) == 3 and parts[1].isascii() and parts[1].isdigit()
                and parts[2] == "kB"):
            try:
                return int(parts[1]) * 1024, ()
            except ValueError:
                # Conversion-limit failures are unknown, never approximated.
                # Do not relax the interpreter's integer-string safety limit.
                return None, (
                    f"MemAvailable exceeds this interpreter's safe "
                    f"integer-conversion limit ({len(parts[1])} "
                    f"digits): unknown, never approximated",)
        return None, (
            f"MemAvailable missing/malformed or not in its documented "
            f"kB unit (expected '<integer> kB'): {line!r}",)
    return None, ("meminfo has no MemAvailable line",)


def _refuse(*notes: str) -> tuple[str, None, None, None, tuple[str, ...]]:
    return CGROUP_UNKNOWN, None, None, None, notes


def _tightest_visible_level(
        levels: list[tuple[Path, int, int]]) -> tuple[Path, int, int]:
    """Pure selection over the visible finite levels seen by the walk: the
    level with the smallest ``limit - current`` (first seen wins a tie,
    i.e. the deepest). ``memory.current`` covers a whole subtree, so a
    parent's usage — sibling usage included — tightens the child. This is
    the tightest VISIBLE bound only: a diagnostic selection, never a
    proven complete budget (see the module docstring's scope gate)."""
    return min(levels, key=lambda lc: max(0, lc[1] - lc[2]))


def _visible_available(levels: list[tuple[Path, int, int]]) -> int | None:
    """Advisory visible availability in bytes (the tightest seen level's
    ``limit - current``) — an UPPER bound on true availability, since any
    hidden constraint can only tighten it — or ``None`` if none were
    seen."""
    if not levels:
        return None
    _, limit, current = _tightest_visible_level(levels)
    return max(0, limit - current)


def _seen_suffix(finite_levels: list[tuple[Path, int, int]]) -> str:
    """Truthful note of finite caps already observed before a refusal."""
    return (", though finite caps were already seen at: "
            + ", ".join(str(d) for d, _, _ in finite_levels)
            if finite_levels
            else ", and no finite cap was found on the visible path")


def read_cgroup_budget(proc_cgroup_path: str | Path = "/proc/self/cgroup",
                       cgroup_fs_root: str | Path = "/sys/fs/cgroup",
                       ) -> tuple[str, int | None, int | None, int | None,
                                  tuple[str, ...]]:
    """Locate this process's cgroup v2 memory constraint by walking **every
    visible ancestor** from the leaf up to and including the mounted root.

    The walk is complete but **advisory**: a descendant cannot escape an
    ancestor's cap, so the tightest ``limit - current`` over every finite
    level seen upper-bounds the applicable availability (each level's
    ``memory.current`` covers its whole subtree, so parent/sibling usage
    tightens the child; an explicit ``max`` at one level does not end the
    walk). Notes name every level's source path and the chosen (tightest)
    level. But because the visible root's COMPLETENESS cannot be proven —
    a cgroup namespace ``/`` can be a non-root cgroup under a tighter
    hidden ancestor, and no controller/mount evidence is read here — the
    real read path returns ``CGROUP_UNKNOWN`` whenever the walk completes,
    **regardless of the visible root's ``memory.max`` contents** (fail-
    closed scope gate, see the module docstring). A finite or unlimited
    state therefore arises only from trusted snapshot injection; the
    returned ``advisory`` number is refusal/diagnostic evidence, never an
    authoritative budget.

    Anything that cannot rule out a tighter constraint also refuses: no
    ``0::`` line (v1-only state is NOT assumed unlimited),
    malformed/ambiguous proc cgroup paths (relative, ``.``, ``..``,
    duplicate ``0::`` records — never joined into the supplied root),
    unreadable/unparseable/negative files, a finite cap without its usage
    counter, and a **missing ``memory.max`` at ANY visible level** —
    ``ENOENT`` does not prove the memory controller is disabled (the
    directory may be missing or mismapped), so missing files refuse at
    every level, never silently skipped or asserted disabled. A finite
    ``memory.max`` of 0 is a valid zero bound.
    """
    try:
        lines = Path(proc_cgroup_path).read_text(encoding="utf-8",
                                                 errors="replace").splitlines()
    except (OSError, ValueError) as exc:
        return _refuse(f"proc cgroup unreadable: {exc}")
    cpaths = [parts[2] for line in lines
              for parts in [line.split(":", 2)]
              if len(parts) == 3 and parts[0] == "0" and parts[1] == ""]
    if not cpaths:
        return _refuse("no cgroup v2 (0::) line in the proc cgroup file: a "
                       "memory cap cannot be ruled out")
    if len(cpaths) > 1:
        return _refuse(f"ambiguous proc cgroup file: {len(cpaths)} distinct "
                       f"'0::' records ({cpaths!r}); the cgroup path cannot "
                       f"be resolved")
    cpath = cpaths[0]
    if not cpath.startswith("/"):
        return _refuse(f"relative cgroup path {cpath!r}: refusing to join a "
                       f"path that could escape the supplied cgroup root")
    names = cpath.split("/")
    if any(name in (".", "..") for name in names):
        return _refuse(f"cgroup path {cpath!r} contains '.' or '..' "
                       f"components: refusing to join a path that could "
                       f"escape the supplied cgroup root")
    names = [name for name in names if name]
    root = Path(cgroup_fs_root)

    notes: list[str] = []
    finite_levels: list[tuple[Path, int, int]] = []
    try:
        for depth in range(len(names), -1, -1):
            directory = root.joinpath(*names[:depth])
            max_file = directory / "memory.max"
            try:
                raw = max_file.read_text(encoding="utf-8",
                                         errors="replace").strip()
            except FileNotFoundError:
                if depth == 0:
                    # The mounted root itself lacks memory.max: it may be
                    # the (uncappable) true root cgroup OR a namespace
                    # boundary hiding capped ancestors — undecidable
                    # without a broad mount parser, so fail closed.
                    return (CGROUP_UNKNOWN, None, None,
                            _visible_available(finite_levels),
                            tuple(notes) + (
                                f"no memory.max at the visible cgroup root "
                                f"{root}: the kernel omits memory.max on the "
                                f"true root cgroup and a cgroup namespace "
                                f"'/' can hide capped ancestors, so an "
                                f"unconstrained hierarchy cannot be "
                                f"proven{_seen_suffix(finite_levels)}",))
                # ENOENT at ANY visible level is unknown: it does not
                # prove the memory controller is disabled there (the
                # directory may be missing or mismapped and no controller
                # evidence was read), so refuse instead of skipping.
                return (CGROUP_UNKNOWN, None, None,
                        _visible_available(finite_levels),
                        tuple(notes) + (
                            f"no memory.max at cgroup level {directory}: a "
                            f"missing memory.max does not prove the memory "
                            f"controller is disabled (the level may be "
                            f"missing or mismapped and no controller "
                            f"evidence was read), so a cap cannot be ruled "
                            f"out{_seen_suffix(finite_levels)}",))
            except (OSError, ValueError) as exc:
                return _refuse(f"memory.max unreadable at {directory}: {exc}")
            if raw == "max":
                notes.append(f"cgroup level {directory}: memory.max "
                             f"explicitly unlimited at this level (ancestor "
                             f"caps still checked)")
                continue
            if not raw.isascii() or not raw.isdigit():
                return _refuse(f"memory.max at {directory} is not an unsigned ASCII integer "
                               f"or 'max' (negative/sign-prefixed/noncanonical counter): {raw!r}")
            try:
                limit = int(raw)
            except ValueError:
                return _refuse(f"memory.max at {directory} is neither an "
                               f"integer nor 'max': {raw!r}")
            try:
                cur_raw = (directory / "memory.current").read_text(
                    encoding="utf-8", errors="replace").strip()
                if not cur_raw.isascii() or not cur_raw.isdigit():
                    return _refuse(f"memory.current at {directory} is not an unsigned ASCII integer "
                                   f"(negative/sign-prefixed/noncanonical counter): {cur_raw!r}")
                current = int(cur_raw)
            except FileNotFoundError:
                return _refuse(
                    f"memory.max present at {directory} but memory.current "
                    f"missing: usage unknown against a finite cap (never "
                    f"assumed zero)")
            except (OSError, ValueError) as exc:
                return _refuse(f"memory.current unreadable/unparseable at "
                               f"{directory} against a finite cap: "
                               f"{exc or cur_raw!r}")
            notes.append(f"cgroup level {directory}: finite cap {limit} B, "
                         f"current {current} B "
                         f"(available {max(0, limit - current)} B)")
            finite_levels.append((directory, limit, current))
    except OSError as exc:  # defensive: path handling on this platform
        return _refuse(f"cgroup filesystem walk failed: {exc}")

    # Completed walk: report the tightest visible bound truthfully, then
    # apply the scope gate — root completeness is unproven whatever the
    # visible root's memory.max contained, so real autodiscovery refuses.
    advisory = _visible_available(finite_levels)
    if finite_levels:
        chosen_dir, limit, current = _tightest_visible_level(finite_levels)
        notes.append(
            f"advisory visible availability {advisory} B: minimum "
            f"limit-minus-current over {len(finite_levels)} visible finite "
            f"level(s); chosen {chosen_dir} (limit {limit} B, current "
            f"{current} B) — an UPPER bound on true availability (hidden "
            f"ancestors can only tighten it), never an authoritative "
            f"complete budget")
    notes.append(_SCOPE_GATE_REASON)
    return CGROUP_UNKNOWN, None, None, advisory, tuple(notes)


def read_memory_snapshot(*, meminfo_path: str | Path = "/proc/meminfo",
                         proc_cgroup_path: str | Path = "/proc/self/cgroup",
                         cgroup_fs_root: str | Path = "/sys/fs/cgroup",
                         ) -> MemorySnapshot:
    """Read real state. The cgroup component is fail-closed ``unknown`` by
    design (scope gate): the walk's per-level notes and the advisory
    visible availability observation travel with the snapshot as
    diagnostic evidence, never as a proven complete budget."""
    host, host_notes = read_host_available(meminfo_path)
    state, limit, current, advisory, notes = read_cgroup_budget(
        proc_cgroup_path, cgroup_fs_root)
    if state != CGROUP_FINITE:
        limit = current = None  # a non-finite state carries no counters
    return MemorySnapshot(host_available_bytes=host, cgroup_state=state,
                          cgroup_limit_bytes=limit, cgroup_current_bytes=current,
                          notes=tuple(host_notes) + tuple(notes),
                          advisory_visible_available_bytes=advisory)


# ---------------------------------------------------------------------------
# Required bytes + verdict
# ---------------------------------------------------------------------------


def _dtype_bytes(name: Any, at: str) -> int:
    if not isinstance(name, str) or name.strip().lower() not in DTYPE_BYTES:
        raise MemoryPreflightError(
            f"{at}: must be a dtype this preflight models "
            f"({', '.join(sorted(set(DTYPE_BYTES)))}), got {name!r}")
    return DTYPE_BYTES[name.strip().lower()]


def encoder_memory_required(*, checkpoint_bytes: int, checkpoint_dtype: str,
                            target_dtype: str, headroom_bytes: int,
                            ) -> tuple[int, int]:
    """``(weights_bytes_in_ram, required_total)`` — an explicit, conservative
    bound. Expansion is ``checkpoint_bytes * target_width / storage_width``
    rounded **up by integer ceiling division** (exact arbitrary-precision
    integer arithmetic: no float conversion, so huge integer inputs neither
    overflow nor lose the conservative bound); an apparent *shrink* is
    clamped to the checkpoint total (the bytes on disk must be read and
    mapped whatever the target dtype); the explicit runtime headroom is
    added on top. There is no implicit headroom default: the caller states
    it."""
    if not _positive_int(checkpoint_bytes):
        raise MemoryPreflightError(
            f"checkpoint_bytes: must be a positive integer, got "
            f"{checkpoint_bytes!r}")
    if not _nonneg_int(headroom_bytes):
        raise MemoryPreflightError(
            f"headroom_bytes: must be an explicit non-negative integer "
            f"(no implicit default), got {headroom_bytes!r}")
    storage = _dtype_bytes(checkpoint_dtype, "checkpoint_dtype")
    target = _dtype_bytes(target_dtype, "target_dtype")
    expanded = -(-checkpoint_bytes * target // storage)
    weights = max(checkpoint_bytes, expanded)
    return weights, weights + headroom_bytes


@dataclass(frozen=True)
class MemoryPreflightReport:
    """The preflight outcome as a JSON-safe record. ``decision`` is
    ``"proceed"`` (a conservative GO — NOT measured fit: passing alone never
    proves the load fits) or ``"refuse"`` (the numbers presented do not fit
    the applicable budget, or the constrained state is unknown).
    ``cgroup_limit_bytes``/``cgroup_current_bytes`` are the **chosen**
    (tightest visible) level's counters; the walk's per-level source paths
    are in ``reasons``."""

    schema: ClassVar[str] = MEM_PREFLIGHT_SCHEMA

    checkpoint_bytes: int
    checkpoint_dtype: str
    target_dtype: str
    dtype_expanded_bytes: int
    headroom_bytes: int
    required_bytes: int
    host_available_bytes: int | None
    cgroup_state: str
    cgroup_limit_bytes: int | None
    cgroup_current_bytes: int | None
    cgroup_available_bytes: int | None
    budget_bytes: int | None
    decision: str
    reasons: tuple[str, ...]
    #: Always True: passing is a conservative prerequisite, never measured
    #: model-fit evidence (see module docstring).
    conservative_precondition: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "checkpoint_bytes": self.checkpoint_bytes,
                "checkpoint_dtype": self.checkpoint_dtype,
                "target_dtype": self.target_dtype,
                "dtype_expanded_bytes": self.dtype_expanded_bytes,
                "headroom_bytes": self.headroom_bytes,
                "required_bytes": self.required_bytes,
                "host_available_bytes": self.host_available_bytes,
                "cgroup_state": self.cgroup_state,
                "cgroup_limit_bytes": self.cgroup_limit_bytes,
                "cgroup_current_bytes": self.cgroup_current_bytes,
                "cgroup_available_bytes": self.cgroup_available_bytes,
                "budget_bytes": self.budget_bytes,
                "decision": self.decision,
                "reasons": list(self.reasons),
                "conservative_precondition": self.conservative_precondition}


def encoder_memory_preflight(checkpoint_bytes: int, *,
                             checkpoint_dtype: str, target_dtype: str,
                             headroom_bytes: int,
                             snapshot: MemorySnapshot | None = None,
                             meminfo_path: str | Path = "/proc/meminfo",
                             proc_cgroup_path: str | Path = "/proc/self/cgroup",
                             cgroup_fs_root: str | Path = "/sys/fs/cgroup",
                             ) -> MemoryPreflightReport:
    """Evaluate the preflight. Tests inject ``snapshot`` or the paths; the
    real call reads only the documented proc/cgroup files. Never reads a
    model file, never imports an ML package, never loads anything."""
    weights, required = encoder_memory_required(
        checkpoint_bytes=checkpoint_bytes, checkpoint_dtype=checkpoint_dtype,
        target_dtype=target_dtype, headroom_bytes=headroom_bytes)
    snap = snapshot if snapshot is not None else read_memory_snapshot(
        meminfo_path=meminfo_path, proc_cgroup_path=proc_cgroup_path,
        cgroup_fs_root=cgroup_fs_root)
    if not isinstance(snap, MemorySnapshot):
        raise MemoryPreflightError("snapshot must be a MemorySnapshot or None")
    budget, reasons = snap.budget()
    cgroup_available = None
    if snap.cgroup_state == CGROUP_FINITE:
        cgroup_available = max(0, snap.cgroup_limit_bytes
                               - snap.cgroup_current_bytes)
    if budget is None:
        decision = "refuse"
        reasons = tuple(reasons) + ("no applicable budget can be proven",)
        advisory = snap.advisory_visible_available_bytes
        if advisory is not None and required > advisory:
            # Truthful refusal evidence kept alongside the scope-gate
            # refusal: the allocation LOWER bound (required) exceeds the
            # visible availability UPPER bound, and hidden ancestors can
            # only tighten availability — so refusal is guaranteed no
            # matter what the hidden state is.
            reasons += (
                f"required {required} B exceeds the tightest visible "
                f"availability {advisory} B: the allocation lower bound is "
                f"above the visible availability upper bound, and hidden "
                f"ancestors can only tighten availability — refusal is "
                f"guaranteed regardless of the hidden state",)
    elif required > budget:
        decision = "refuse"
        reasons = tuple(reasons) + (
            f"required {required} B exceeds the applicable budget "
            f"{budget} B",)
    else:
        decision = "proceed"
        reasons = tuple(reasons) + (
            "conservative prerequisite satisfied only: this does NOT prove "
            "the load fits (peak residency, allocator and conversion "
            "transients are not modelled)",)
    return MemoryPreflightReport(
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_dtype=checkpoint_dtype, target_dtype=target_dtype,
        dtype_expanded_bytes=weights, headroom_bytes=headroom_bytes,
        required_bytes=required,
        host_available_bytes=snap.host_available_bytes,
        cgroup_state=snap.cgroup_state,
        cgroup_limit_bytes=snap.cgroup_limit_bytes,
        cgroup_current_bytes=snap.cgroup_current_bytes,
        cgroup_available_bytes=cgroup_available,
        budget_bytes=budget, decision=decision, reasons=reasons)


def require_encoder_memory_fit(checkpoint_bytes: int, **kwargs: Any
                               ) -> MemoryPreflightReport:
    """:func:`encoder_memory_preflight` that RAISES
    :class:`MemoryPreflightError` instead of returning a refusal — the
    loader-side call site fails typed, before any ML import, weight read or
    loader call, on the first sign the numbers do not fit."""
    report = encoder_memory_preflight(checkpoint_bytes, **kwargs)
    if report.decision != "proceed":
        raise MemoryPreflightError(
            "encoder memory preflight refused: "
            + "; ".join(report.reasons)
            + f"; report={report.to_dict()}")
    return report
