"""Default local loader + Qwen-Image-2.1 diffusers adapter (M4 groundwork).

Boundary rules:

* Importing this module stays **ML-free**. ``torch``, ``diffusers`` and
  ``transformers`` are imported only inside :meth:`Qwen21DiffusersAdapter.load`,
  i.e. only on an actual run.
* Every weight comes from the verified local reference bundle
  (``manifest`` + ``models_dir`` -> :func:`resolve_local_bundle`), never from a
  repo id, an ambient Hugging Face cache or the network. The CLI has already
  resolved the same bundle once; the adapter re-resolves so it never trusts a
  caller-supplied path list, and pays the documented re-hash cost.
* Only the reviewed resident profile is implemented: unquantized FP16 on one
  V100, native SDPA, transformer + VAE resident, no text encoder in-process
  (the Qwen3-VL encoder runs in the subprocess worker), no hooks/offload.
  Q8/Q4 and streaming are M5 work and are refused here rather than silently
  ignored.
* The adapter **loads and attaches**; it grants no memory fit, no calibration
  and no image-quality acceptance. The measured memory policy (M5.3) remains a
  separate gate, and the CPU encoder's own conservative host-memory preflight
  still refuses until complete trusted budget evidence exists. The single
  opt-in exception is the constructor's ``trust_unbounded_host`` — the
  operator's own attestation that this process tree has no cgroup memory
  limit; it is off by default and the gate still applies a fresh
  ``MemAvailable`` read, so it can never supply a number.
* Nothing here writes outside the configured cache/output directories, changes
  a driver, touches ROCm, or flushes a global allocator on behalf of other
  processes.
"""
from __future__ import annotations

import gc
import threading
from functools import partial
from pathlib import Path
from typing import Any

from qwen_engine.adapters.supplied_qwen21 import SuppliedQwen21Adapter
from qwen_engine.config import (
    AttentionBackend,
    ConfigError,
    FieldError,
    GenerationConfig,
    OffloadMode,
    Quant,
)
from qwen_engine.encoder import EmbeddingCache
from qwen_engine.encoder_client import ProductionEncoder
from qwen_engine.errors import EngineError, HardwareAbsentError
from qwen_engine.local_bundle import LocalReferenceBundle, resolve_local_bundle
from qwen_engine.manifest import load_manifest
from qwen_engine.runtime_identity import observe_generation_runtime

#: The only encoder dtype this loader asks the CPU worker for. The checkpoint is
#: BF16, so this expands no weights; fp16/fp32 are alternatives only after a
#: measured decision (C5).
ENCODER_DTYPE = "bf16"

#: Explicit, non-defaulted host headroom passed to the encoder worker's
#: conservative preflight. It is a *prerequisite* input, not a measured peak.
ENCODER_HEADROOM_BYTES = 4 * 1024**3

#: Bounded encoder worker lifetime; a full 17.5 GB CPU load is slow.
ENCODER_TIMEOUT_S = 3600.0

#: The reviewed execution device is the first CUDA logical device. The
#: hardware gate below still verifies its identity before anything is loaded.
LOGICAL_DEVICE_INDEX = 0

#: Latent-token count above which the transformer's SwiGLU MLP is evaluated in
#: token chunks (1024² is exactly 4096, so 1024² and below stay unchunked).
#: At 2048² the joint sequence is 16,384 tokens and one SwiGLU intermediate is
#: ~386 MiB; two of them are live at once, which is what pushed the measured
#: 2K attempt over the 16 GiB card. Chunking is exact — the MLP is token-wise —
#: so it changes peak memory and kernel count, never the result.
MLP_CHUNK_ABOVE_TOKENS = 4096

#: Token chunk used above that threshold. 2048 tokens keeps each intermediate
#: near 48 MiB and measured 2K denoise inside the card with headroom.
MLP_CHUNK_TOKENS = 2048

#: Tiled-decode geometry. The library default (256 px tiles, 192 px stride)
#: leaves a visible grid of dashed seams on a 2048² image; 1024 px tiles with
#: a 768 px stride leave one boundary per axis and measured clean at 1:1.
TILED_DECODE_TILE = 1024
TILED_DECODE_STRIDE = 768


def _chunked_mlp_forward(inner_forward, torch, chunk: int, hidden_states):
    """Evaluate a token-wise MLP in fixed token chunks (exact, not an approximation)."""
    tokens = hidden_states.shape[1]
    if tokens <= chunk:
        return inner_forward(hidden_states)
    parts = [inner_forward(hidden_states[:, start:start + chunk])
             for start in range(0, tokens, chunk)]
    return torch.cat(parts, dim=1)


def _apply_mlp_chunking(torch, transformer, chunk: int) -> int:
    """Bound the SwiGLU intermediates for high token counts.

    Only the per-block image MLP is touched: it is token-wise, so slicing the
    sequence dimension is mathematically identical to one pass. Text tokens do
    not pass through it. Returns the number of wrapped blocks for the caller's
    evidence; a model without the expected blocks refuses rather than silently
    running unchunked.
    """
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks:
        raise EngineError("transformer layout has no blocks to chunk")
    wrapped = already = 0
    for block in blocks:
        mlp = getattr(block, "img_mlp", None)
        if mlp is None:
            continue
        if getattr(mlp, "_token_chunk", None) is not None:
            already += 1
            continue
        mlp._token_chunk = chunk
        mlp.forward = partial(_chunked_mlp_forward, mlp.forward, torch, chunk)
        wrapped += 1
    if wrapped + already != len(blocks):
        raise EngineError(
            f"MLP chunking wrapped {wrapped} of {len(blocks)} transformer blocks")
    return wrapped


def _libraries():
    """Import the heavy stack *and* validate the pinned runtime first.

    Package policy runs before every ML import, exactly like the supplied
    component bridge; a drifted environment refuses here rather than loading
    mismatched classes.
    """
    from qwen_engine.encoder_identity import (
        observe_installed_packages,
        validate_installed_packages,
    )

    validate_installed_packages(observe_installed_packages())
    import torch
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from transformers import Qwen3VLProcessor

    return (
        torch,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        Qwen3VLProcessor,
    )


def _require_v100(torch) -> Any:
    """Refuse anything but the reviewed Volta profile before a weight is read.

    Device *enumeration* is not execution proof, but a wrong CUDA build, a
    non-Volta device or an absent sm_70 kernel set cannot run this stack, and
    must surface as the hardware exit code (5) rather than a load traceback.
    """
    if getattr(torch.version, "cuda", None) != "12.6" or not torch.cuda.is_available():
        raise HardwareAbsentError(
            "reviewed CUDA 12.6 PyTorch build and a visible V100 are required; "
            "install the pinned cu126 wheel and check the device"
        )
    if not 0 <= LOGICAL_DEVICE_INDEX < torch.cuda.device_count():
        raise HardwareAbsentError("the reviewed logical CUDA device is not visible")
    properties = torch.cuda.get_device_properties(LOGICAL_DEVICE_INDEX)
    if (
        (properties.major, properties.minor) != (7, 0)
        or "V100" not in properties.name
        or not 15 * 1024**3 <= properties.total_memory <= 16 * 1024**3
    ):
        raise HardwareAbsentError(
            "a Tesla V100 16GB-class device is required; "
            f"found {properties.name!r} (compute {properties.major}.{properties.minor})"
        )
    if "sm_70" not in torch.cuda.get_arch_list():
        raise HardwareAbsentError(
            "the installed torch build has no sm_70 kernels; the pinned cu126 "
            "wheel is required for Volta"
        )
    return properties


def _validate_policy(config: GenerationConfig) -> None:
    """Review-scope refusals, before any manifest read or ML import."""
    if not isinstance(config, GenerationConfig):
        raise ConfigError(FieldError("config", "a GenerationConfig is required"))
    errors: list[FieldError] = []
    if config.quant is not Quant.FP16:
        errors.append(FieldError(
            "quant", "the resident loader currently supports fp16 only; "
                     "q8/q4 belong to the M5 quality ladder"))
    if config.offload is not OffloadMode.RESIDENT:
        errors.append(FieldError(
            "offload", "the resident loader supports offload=resident only; "
                       "streamed/offload modes are M5"))
    if config.attention_backend not in (AttentionBackend.AUTO, AttentionBackend.SDPA):
        errors.append(FieldError(
            "attention_backend", "this loader supports auto/sdpa only; "
                                  "other backends are benchmark work"))
    for name in ("manifest_path", "models_dir"):
        if getattr(config, name) is None:
            errors.append(FieldError(name, "an actual run requires this path"))
    if config.output_dir is None:
        errors.append(FieldError("output_dir", "an actual run requires an output directory"))
    if config.cache_dir is None:
        errors.append(FieldError(
            "cache_dir", "the encoder cache/scratch requires an explicit cache_dir; "
                         "add --cache-dir cache (created automatically) "
                         "or set cache_dir in the JSON config"))
    if errors:
        raise ConfigError(errors)


class Qwen21DiffusersAdapter:
    """One-shot local loader over the reviewed supplied-component bridge.

    ``load`` resolves and verifies the local bundle, checks the pinned runtime
    and the V100 profile, builds the pinned pipeline classes locally (processor,
    transformer, VAE, scheduler; no text encoder), places them as resident FP16
    components, then delegates the five phases to
    :class:`~qwen_engine.adapters.supplied_qwen21.SuppliedQwen21Adapter`.
    """

    def __init__(self, *, trust_unbounded_host: bool = False) -> None:
        if not isinstance(trust_unbounded_host, bool):
            raise ConfigError(
                "trust_unbounded_host must be an explicit boolean operator attestation")
        self._event: threading.Event | None = None
        self._inner: SuppliedQwen21Adapter | None = None
        self._pipeline: Any = None
        self._bundle: LocalReferenceBundle | None = None
        self._torch: Any = None
        self._vae_cls: Any = None
        self._vae_dir: Path | None = None
        self._tiled_decode = False
        self._mlp_chunk_tokens = 0
        self._mlp_chunked_blocks = 0
        self._trust_unbounded_host = trust_unbounded_host
        self._state = "new"
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        if self._state != "new" or (callback is not None and not callable(callback)):
            raise ConfigError("bind a callable progress callback or None before load")
        self._progress_callback = callback

    def set_cancel_event(self, event: threading.Event | None) -> None:
        if self._state != "new" or (event is not None and not isinstance(event, threading.Event)):
            raise ConfigError("bind a threading.Event or None before load")
        self._event = event

    def load(self, config: GenerationConfig, assets) -> None:
        if self._state != "new" or self._inner is not None:
            raise EngineError("the loader adapter is single-use and already attached")
        _validate_policy(config)
        manifest = load_manifest(config.manifest_path)
        bundle = resolve_local_bundle(manifest, config.models_dir)
        bundle_assets = bundle.to_assets()
        # The CLI resolver reports paths in the caller's lexical form (e.g. a
        # relative --models-dir); the bundle publishes canonical absolutes.
        # Compare resolved identities, then attach the canonical set.
        supplied_assets = {
            role: tuple(Path(path).resolve() for path in paths)
            for role, paths in dict(assets).items()
        }
        if supplied_assets != bundle_assets:
            raise ConfigError(FieldError(
                "assets", "verified asset paths disagree with the pinned local bundle"))

        # CPU-side identity before the CUDA context exists; the post-context
        # observation below must bind to this exact device UUID/index.
        observe_generation_runtime()

        (
            torch,
            pipeline_cls,
            transformer_cls,
            vae_cls,
            scheduler_cls,
            processor_cls,
        ) = _libraries()
        properties = _require_v100(torch)
        device = torch.device("cuda", LOGICAL_DEVICE_INDEX)

        try:
            # Load and place one component at a time: the CPU copy of each is
            # released before the next component's weights are read, so host
            # peak stays near one model rather than the whole stack.
            transformer = transformer_cls.from_pretrained(
                bundle.component_dirs["dit"], torch_dtype=torch.float16,
                local_files_only=True)
            transformer.to(device=device, dtype=torch.float16).eval()
            vae = vae_cls.from_pretrained(
                bundle.component_dirs["vae"], torch_dtype=torch.float16,
                local_files_only=True)
            vae.to(device=device, dtype=torch.float16).eval()
            scheduler = scheduler_cls.from_pretrained(
                bundle.component_dirs["scheduler"], local_files_only=True)
            processor = processor_cls.from_pretrained(
                bundle.component_dirs["processor"], local_files_only=True)
        except Exception as exc:
            raise EngineError(
                f"local component load refused: {type(exc).__name__}: {exc}") from exc

        pipeline = pipeline_cls(
            scheduler=scheduler, vae=vae, text_encoder=None,
            processor=processor, transformer=transformer)

        runtime = observe_generation_runtime()
        if not runtime.matches_cuda_device(
                uuid=str(properties.uuid), logical_device_index=LOGICAL_DEVICE_INDEX):
            raise EngineError(
                "generation runtime identity is not bound to the reviewed CUDA device")

        cache_root = Path(config.cache_dir)
        scratch = cache_root / "encoder-scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        encoder = ProductionEncoder(
            manifest_path=config.manifest_path,
            models_dir=config.models_dir,
            scratch_dir=scratch,
            headroom_bytes=ENCODER_HEADROOM_BYTES,
            timeout_s=ENCODER_TIMEOUT_S,
            cache=EmbeddingCache(cache_root / "embeds"),
            attested_unbounded_host=self._trust_unbounded_host,
        )

        inner = SuppliedQwen21Adapter(
            pipeline=pipeline,
            encoder=encoder,
            manifest=manifest,
            bundle=bundle,
            encoder_dtype=ENCODER_DTYPE,
            execution_dtype="fp16",
            use_kv_cache=True,
        )
        inner.set_cancel_event(self._event)
        inner.set_progress_callback(self._progress_callback)
        inner.load(config, bundle_assets)
        if config.latent_tokens > MLP_CHUNK_ABOVE_TOKENS:
            self._mlp_chunked_blocks = _apply_mlp_chunking(
                torch, pipeline.transformer, MLP_CHUNK_TOKENS)
            self._mlp_chunk_tokens = MLP_CHUNK_TOKENS
        self._pipeline = pipeline
        self._bundle = bundle
        self._torch = torch
        self._vae_cls = vae_cls
        self._vae_dir = bundle.component_dirs["vae"]
        self._tiled_decode = config.vae_tiled_decode
        self._inner = inner
        self._state = "loaded"

    def _attached(self) -> SuppliedQwen21Adapter:
        if self._inner is None:
            raise EngineError("loader adapter is not attached; call load() first")
        return self._inner

    def encode(self, config: GenerationConfig):
        return self._attached().encode(config)

    def calibration_facts(self):
        """Delegate the exact conditioning facts to the attached adapter."""
        return self._attached().calibration_facts()

    def denoise(self, config: GenerationConfig, conditioning):
        self._release_vae()
        latents = self._attached().denoise(config, conditioning)
        self._release_denoiser()
        return latents

    def decode(self, latents):
        self._restore_vae()
        return self._attached().decode(latents)

    def _release_vae(self) -> None:
        """Take the VAE off the card while the transformer denoises.

        Phase-exclusive residency: denoise needs only the transformer, decode
        needs only the VAE, so the two never have to be resident together.
        Dropping the object (rather than moving it to CPU) also keeps
        ``_execution_device`` unambiguous — a CPU-placed VAE would otherwise be
        the first module the pipeline inspects. The VAE is reloaded from the
        same verified local directory before decode.
        """
        pipeline = self._pipeline
        if pipeline is None or getattr(pipeline, "vae", None) is None:
            return
        pipeline.vae = None
        gc.collect()

    def _restore_vae(self) -> None:
        pipeline = self._pipeline
        if pipeline is None or self._vae_cls is None or self._vae_dir is None:
            return
        if getattr(pipeline, "vae", None) is not None:
            return
        torch = self._torch
        # Tiled decode runs the VAE in fp32: at 2048² the fp16 tiled path with
        # the library's default 256 px tiles produced non-finite output
        # (measured; the same latents decode finite in fp32), and the
        # transformer is already released, so the wider activations fit
        # (13.4 GiB peak with 1024 px tiles). Untiled decode stays fp16 — it
        # fits the 1024² envelope, and fp16/fp32 differ by at most 1/255 there
        # (2/255 with 2K tiling), so precision is not buying visible quality.
        dtype = torch.float32 if self._tiled_decode else torch.float16
        vae = self._vae_cls.from_pretrained(
            self._vae_dir, torch_dtype=dtype, local_files_only=True)
        vae.to(device=torch.device("cuda", LOGICAL_DEVICE_INDEX),
               dtype=dtype).eval()
        if self._tiled_decode:
            vae.enable_tiling(
                tile_sample_min_height=TILED_DECODE_TILE,
                tile_sample_min_width=TILED_DECODE_TILE,
                tile_sample_stride_height=TILED_DECODE_STRIDE,
                tile_sample_stride_width=TILED_DECODE_STRIDE,
            )
        pipeline.vae = vae

    def _release_denoiser(self) -> None:
        """Drop the transformer's VRAM before the VAE decode (single-use loader).

        The bridge never touches the transformer again after denoise (decode
        uses the VAE, save uses the image), and this adapter is single-use, so
        releasing the reference is safe. The freed blocks stay in this
        process's own allocator cache and are reused by the decode allocation;
        no global allocator flush is performed and nothing outside this process
        is affected.

        ``pipeline.transformer = None`` is not enough on its own: diffusers
        attaches a per-module hook registry that refers back to the module, so
        the transformer survives as an unreachable reference cycle (measured:
        13.9 GiB still allocated after the assignment, 0.64 GiB after
        ``gc.collect()`` breaks the cycle). The collector runs only after the
        reference is dropped, so it cannot collect anything still in use.

        Why it matters (measured on the V100): transformer + VAE resident is
        ~14.8 GiB of 16 GiB, which leaves too little for an untiled 1024²
        decode. Untiled decode at 1024² is also measurably *better* than the
        tiled path: tiled decode leaves faint vertical dashed streaks in flat
        regions (max channel difference 52/255, mean 1.24 against the resident
        decode of the same latents), while the resident decode is clean.
        ``--tiled-decode`` therefore stays available as an explicit
        memory-first knob, not as the default.
        """
        pipeline = self._pipeline
        if pipeline is None or getattr(pipeline, "transformer", None) is None:
            return
        pipeline.transformer = None
        gc.collect()

    def save(self, image, config: GenerationConfig):
        return self._attached().save(image, config)

    def close(self) -> None:
        """Drop owned references; never delete outputs, flush allocators or
        touch another process's devices. Idempotent after partial load."""
        inner, self._inner = self._inner, None
        self._pipeline = None
        self._bundle = None
        self._state = "closed"
        self._progress_callback = None
        if inner is not None:
            inner.close()
