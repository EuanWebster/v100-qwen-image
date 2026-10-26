"""Single-use phase bridge over explicitly supplied, trusted components.

No loader, resource approval, checkpoint provenance or public backend selector.
The caller owns those gates BEFORE transferring resident components here.
Plain import is ML-free; the default CLI factory remains absent/disabled.
"""
from __future__ import annotations

import hashlib
import io
import threading
import uuid
from pathlib import Path

from qwen_engine import reference_store
from qwen_engine.config import (AttentionBackend, ConfigError, GenerationConfig,
                                OffloadMode, Quant, Recipe, TURBO_STEPS)
from qwen_engine.encoder_client import ProductionEncoding
from qwen_engine.encoder import EncoderCancelledError
from qwen_engine.encoder_identity import project_encoder_seed, validate_completed_seed
from qwen_engine.encoder_production import CANONICAL_DTYPES, payload_to_branches
from qwen_engine.errors import EngineError, GenerationCancelled
from qwen_engine.local_bundle import (
    LOCAL_REFERENCE_LAYOUT, ROLE_DIRECTORIES, LocalReferenceBundle,
    reference_bundle_fingerprints,
)
from qwen_engine.manifest import ModelManifest
from qwen_engine.performance import stage, current

MAX_PNG_BYTES = 256 * 1024 * 1024

#: Exact pinned ``QwenImage21Pipeline.encode_prompt`` parameter list (bound).
#: The reference mask hook refuses on ANY drift instead of guessing.
_ENCODE_PROMPT_PARAMETERS = ("prompt", "image", "device", "num_images_per_prompt",
                             "prompt_embeds", "prompt_embeds_mask", "image_pad_mask")

#: Domain-separation label for the reference-edit noise seed (see
#: :func:`derive_edit_noise_seed`). Fixed forever: changing it changes every
#: edit run's noise, i.e. it is part of the generation identity.
EDIT_NOISE_DERIVATION = "qwen-engine.edit-noise.v1"


def derive_edit_noise_seed(seed: int, reference_ids: tuple[str, ...]) -> int:
    """Deterministic 64-bit SAMPLER-noise seed for reference edits.

    Root cause 2026-10-26 (output/gpu/FINDINGS-edit-grid.md): references
    produced by this engine at 1024x1024/seed 42 edited at the same size and
    seed start from the IDENTICAL initial noise tensor that generated the
    reference, injecting its content twice (conditioning + starting noise) —
    the over-processed artifact. Deriving the edit noise seed from
    ``sha256(label || seed || ordered reference ids)`` makes that collision
    structurally impossible: the edit noise can never equal the plain
    ``seed``-noise of any generation. Purely domain separation — T2I keeps
    using ``config.seed`` exactly, and the user-visible seed/filename stay
    ``config.seed``; only the sampler's manual_seed argument changes.
    Ids are length-prefixed so concatenation cannot be forged, and order is
    significant (it is part of the conditioning identity).
    """
    digest = hashlib.sha256()
    digest.update(EDIT_NOISE_DERIVATION.encode("utf-8"))
    digest.update(int(seed).to_bytes(8, "big"))
    for reference_id in reference_ids:
        raw = str(reference_id).encode("utf-8")
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
    return int.from_bytes(digest.digest()[:8], "big")


def load_reference_images(config, image_module):
    """Hash-verified reference PILs in config order (module-level so the
    loader can use it BEFORE the bridge is attached; ``image_module`` is the
    caller's PIL.Image — the injection seam the supplied-adapter tests use).

    Bytes come through ``reference_store.read_blob`` (bounded read +
    sha256 == id recheck), so a blob replaced mid-run fails closed before
    any decode. Decoding is capped: the declared side lengths are checked
    against the store side bound first, then the pixels are loaded (PIL's
    own decompression-bomb guard is the pixel-count backstop). Both views
    of a run — the loader's load-phase reference VAE encode and this
    adapter's ``reference_images`` — read the same store ids, so identical,
    re-verified bytes are what each sees.
    """
    references = tuple(getattr(config, "references", ()))
    directory = getattr(config, "references_dir", None)
    if not references or directory is None:
        raise EngineError("reference loading requested without ids or the store directory")
    images = []
    for position, reference_id in enumerate(references, start=1):
        raw = reference_store.read_blob(directory, reference_id)
        try:
            image = image_module.open(io.BytesIO(raw))
            width, height = image.size
            if (not 1 <= width <= reference_store.MAX_REFERENCE_DIM
                    or not 1 <= height <= reference_store.MAX_REFERENCE_DIM):
                raise EngineError(
                    f"reference {position} decodes to {width}x{height}; each side "
                    f"must not exceed {reference_store.MAX_REFERENCE_DIM} pixels")
            image.load()
        except EngineError:
            raise
        except Exception:
            raise EngineError(
                f"reference {position} cannot be decoded as an image") from None
        images.append(image)
    return images


def _libraries():
    from qwen_engine.encoder_identity import observe_installed_packages, validate_installed_packages
    validate_installed_packages(observe_installed_packages())
    import torch
    from PIL import Image
    from diffusers import (QwenImage21Pipeline, QwenImage21Transformer2DModel,
                           AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler)
    return torch, Image, (QwenImage21Pipeline, QwenImage21Transformer2DModel,
                          AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler)


class _BoundedPNG(io.BytesIO):
    def write(self, value):
        if max(self.tell() + len(value), len(self.getbuffer())) > MAX_PNG_BYTES:
            raise EngineError("PNG exceeds the artifact byte cap")
        return super().write(value)


class SuppliedQwen21Adapter:
    """Attach an approved resident pipeline; never load/move its models.

    The encoder provider follows ProductionEncoder.encode. Explicit supplied
    components are a trusted Python API, not proof they came from the bundle.
    Execution dtype is separate from encoder dtype. fp32 supports CPU component
    witnesses, NOT a new approved hardware/memory profile. No hooks/offload or
    quantized models are supported here. Calls and close must be serial.
    """
    def __init__(self, *, pipeline, encoder, manifest: ModelManifest,
                 bundle: LocalReferenceBundle, encoder_dtype: str,
                 execution_dtype: str = "fp16", use_kv_cache: bool = True):
        if (encoder_dtype not in CANONICAL_DTYPES
                or execution_dtype not in ("fp16", "fp32")
                or type(use_kv_cache) is not bool):
            raise ConfigError("explicit canonical dtypes and a boolean KV policy are required")
        if not isinstance(manifest, ModelManifest) or not isinstance(bundle, LocalReferenceBundle):
            raise ConfigError("explicit manifest and local reference bundle are required")
        fingerprints = reference_bundle_fingerprints(manifest)
        expected = {role: tuple(bundle.root / name for name in names)
                    for role, names in LOCAL_REFERENCE_LAYOUT.items()}
        if (bundle.fingerprints != fingerprints or bundle.to_assets() != expected
                or dict(bundle.component_dirs) != {
                    role: bundle.root / name for role, name in ROLE_DIRECTORIES.items()}):
            raise ConfigError("supplied bundle disagrees with the manifest/layout")
        if not callable(getattr(encoder, "encode", None)):
            raise ConfigError("an explicit encoder provider is required")
        self._pipeline, self._encoder = pipeline, encoder
        self._manifest, self._bundle = manifest, bundle
        self.encoder_dtype, self.execution_dtype = encoder_dtype, execution_dtype
        self.use_kv_cache = use_kv_cache
        self._state, self._event, self._config = "new", None, None
        self._torch = self._image = self._device = self._dtype = None
        self._encoding_request = None
        self._conditioning = None
        self.actual_seed = None
        self._progress_callback = None
        self._reference_pils = None
        self._noise_seed = None

    def set_progress_callback(self, callback):
        if self._state != "new" or (callback is not None and not callable(callback)):
            raise ConfigError("bind a callable progress callback or None before attachment")
        self._progress_callback = callback

    def _check(self, state, config=None):
        if self._state != state:
            raise EngineError("supplied adapter phase order/ownership violation")
        if config is not None and config != self._config:
            raise ConfigError("generation config changed after attachment")
        if self._event is not None and self._event.is_set():
            raise GenerationCancelled("supplied generation cancelled before publication")

    def set_cancel_event(self, event):
        if self._state != "new" or (event is not None and not isinstance(event, threading.Event)):
            raise ConfigError("bind a threading.Event or None before attachment")
        self._event = event

    def load(self, config: GenerationConfig, assets):
        self._check("new")
        if not isinstance(config, GenerationConfig):
            raise ConfigError("GenerationConfig is required")
        if config.quant is Quant.Q8 and config.q8_manifest_path is None:
            raise ConfigError("Q8 attachment requires the explicit sidecar identity")
        if (config.quant not in (Quant.FP16, Quant.Q8) or config.offload != OffloadMode.RESIDENT
                or config.attention_backend not in (AttentionBackend.AUTO, AttentionBackend.SDPA)):
            raise ConfigError("supplied resident adapter supports FP16 or reviewed Q8_0 with native SDPA")
        if assets != self._bundle.to_assets():
            raise ConfigError("attached assets disagree with the supplied bundle")
        torch, image, classes = _libraries()
        pipe = self._pipeline
        if type(pipe) is not classes[0]:
            raise EngineError("supplied pipeline must be the exact pinned Qwen21 class")
        # A reference run does not require the VAE resident at load time:
        # the loader computed the reference latents during load
        # (reference_vae_encode), dropped the VAE before the transformer
        # was loaded, and restores a fresh one for decode with its own
        # tiling policy. Every other run demands full exact placement.
        references = bool(tuple(getattr(config, "references", ())))
        if references:
            components = (pipe, pipe.transformer, pipe.scheduler)
            expected = (classes[0], classes[1], classes[3])
        else:
            components = (pipe, pipe.transformer, pipe.vae, pipe.scheduler)
            expected = classes
        if any(type(value) is not cls for value, cls in zip(components, expected)):
            raise EngineError("supplied components must be the exact pinned Qwen21 classes")
        if pipe.text_encoder is not None or getattr(pipe, "_all_hooks", None):
            raise EngineError("resident generation must not retain an encoder/offload hooks")
        dtype = {"fp16": torch.float16, "fp32": torch.float32}[self.execution_dtype]
        device = pipe._execution_device
        if self.execution_dtype == "fp32" and str(device).split(":", 1)[0] != "cpu":
            raise EngineError("fp32 supplied execution is CPU-witness-only")
        for model in ((pipe.transformer,) if references else (pipe.transformer, pipe.vae)):
            if any(getattr(module, "_hf_hook", None) is not None for module in model.modules()):
                raise EngineError("resident components must not have dispatch/offload hooks")
            if model is pipe.transformer and config.quant is Quant.Q8:
                from qwen_engine.adapters.q8_loader import validate_q8_parameters
                validate_q8_parameters(model, device, dtype)
            elif any(p.is_meta or p.device != device or p.dtype != dtype for p in model.parameters()):
                raise EngineError("supplied model placement/dtype disagrees with execution policy")
            if model.training:
                raise EngineError("supplied components must already be in evaluation mode")
        # latent_channels is projected from the live VAE's z_dim at pipeline
        # construction, so a non-64 z_dim fails here even when the VAE has
        # already left the card for a reference run.
        if (pipe.vae_scale_factor != 16 or pipe.latent_channels != 64
                or pipe.transformer.config.in_channels != 64
                or pipe.transformer.config.out_channels != 64
                or (pipe.vae is not None and pipe.vae.config.z_dim != 64)):
            raise EngineError("unsupported latent architecture")
        pipe.transformer.set_attention_backend("native")
        if pipe.vae is not None:
            # Only meaningful for a still-resident VAE; the decode VAE is
            # restored fresh by the loader with its own tiling policy.
            if config.vae_tiled_decode:
                pipe.vae.enable_tiling()
            else:
                pipe.vae.disable_tiling()
        self._torch, self._image, self._device, self._dtype = torch, image, device, dtype
        self._config, self._state = config, "loaded"

    def encode(self, config):
        self._check("loaded", config)
        # config.references are the ordered identity binding; the encoder
        # child re-hashes every path it reads against them. Paths come from
        # the same reference store the loader below reads (no filesystem
        # probe here: path_for is syntax-only, like the worker IPC side).
        image_files: tuple[str, ...] = ()
        if config.references:
            if config.references_dir is None:
                raise EngineError("reference conditioning requires the "
                                  "reference store directory (references_dir)")
            image_files = tuple(str(reference_store.path_for(config.references_dir, reference_id))
                                for reference_id in config.references)
        try:
            encoding = self._encoder.encode(config, encoder_dtype=self.encoder_dtype,
                                            cancel_event=self._event,
                                            image_files=image_files)
        except EncoderCancelledError:
            # A provider's cancellation acknowledgement is only valid when
            # THIS job requested it; never relabel unrelated provider failures.
            self._check("loaded", config)
            raise
        self._check("loaded", config)
        if not isinstance(encoding, ProductionEncoding):
            raise EngineError("encoder must return a validated ProductionEncoding")
        seed = project_encoder_seed(config, self._manifest, encoder_dtype=self.encoder_dtype)
        validate_completed_seed(seed, encoding.request, self._manifest)
        branches = payload_to_branches(encoding.payload, expect_request=encoding.request)
        if branches.positive.embeds.shape[-1] != self._pipeline.transformer.config.context_in_dim:
            raise EngineError("conditioning hidden width disagrees with the transformer")
        self._encoding_request = encoding.request
        self._conditioning = branches
        # Noise-seed domain separation for reference edits (see
        # derive_edit_noise_seed): decided at ENCODE time so the calibration
        # collector — which binds facts when the denoise phase opens, before
        # denoise runs — already sees it. A seedless run (config.seed is
        # None) draws its seed inside denoise; the noise seed is derived
        # there and the facts bound earlier carry None.
        self._noise_seed = (derive_edit_noise_seed(config.seed, tuple(config.references))
                            if config.references and config.seed is not None else None)
        self._state = "encoded"
        return branches

    def calibration_facts(self):
        """Exact run facts the measured-memory collector must record.

        Read only after encoding: the token lengths come from the validated
        conditioning tensors that were actually processed (padded sequence
        length, i.e. the memory-relevant count), and ``cfg_active`` is the
        config's true-CFG predicate. ``embedding_cache_hit`` is the encoder's
        own last decision; ``None`` (unmeasured) makes the collector refuse.
        """
        self._check("encoded")
        branches = self._conditioning
        if branches is None:
            raise EngineError("no conditioning is attached")
        negative = branches.negative
        return {
            "prompt_tokens": int(branches.positive.embeds.shape[1]),
            "negative_prompt_tokens": (
                int(negative.embeds.shape[1]) if negative is not None else None),
            "cfg_active": bool(self._config.cfg_active),
            "encoder_dtype": self.encoder_dtype,
            "execution_dtype": self.execution_dtype,
            "use_kv_cache": bool(self.use_kv_cache),
            "embedding_cache_hit": getattr(self._encoder, "last_cache_hit", None),
            "references": len(tuple(getattr(self._config, "references", ()))),
            "recipe": self._config.recipe.value,
            # Noise-seed domain separation: the value the sampler is actually
            # seeded with on an edit run (None for T2I/seedless runs, which
            # keep using config.seed itself).
            "noise_seed": self._noise_seed,
            "noise_seed_derivation": (EDIT_NOISE_DERIVATION
                                      if self._noise_seed is not None else None),
        }

    def reference_images(self, config):
        """Hash-verified reference PILs in config order, loaded once per attachment.

        Bytes come through ``reference_store.read_blob`` (bounded read +
        sha256 == id recheck), so a blob replaced mid-run fails closed before
        any decode. Decoding is capped: the declared side lengths are checked
        against the store side bound first, then the pixels are loaded (PIL's
        own decompression-bomb guard is the pixel-count backstop). The list is
        handed to the pipeline as ``image=`` verbatim: the pinned pipeline
        performs its own resize for both the VL and the VAE path (design §1.1).
        """
        self._check("encoded", config)
        if self._reference_pils is None:
            self._reference_pils = load_reference_images(config, self._image)
        return self._reference_pils

    def _install_image_pad_mask_hook(self, pairs):
        """Install a per-call instance override of the pinned ``encode_prompt``.

        ``__call__`` never forwards ``image_pad_mask`` and refuses cached
        embeds that cover condition images without one, while the pinned
        ``encode_prompt`` itself accepts it (design §1.6 amendment). The
        override matches the positive/negative embedding tensors by identity —
        the box holds strong references so ids cannot be recycled mid-call —
        injects the cached mask for the matching branch and delegates
        unchanged to the original bound method. The returned closure removes
        the override and empties the box, and the caller's ``finally`` runs it
        on every path, so nothing pins frames or tensors into later memory
        measurement. Signature drift or a stale override is a refusal, never
        a guess.
        """
        import inspect

        pipe = self._pipeline
        if "encode_prompt" in vars(pipe):
            raise EngineError("an encode_prompt override is already installed")
        original = pipe.encode_prompt
        if tuple(inspect.signature(original).parameters) != _ENCODE_PROMPT_PARAMETERS:
            raise EngineError("pinned pipeline encode_prompt signature has changed; "
                              "cached reference masks cannot be injected blindly")
        box = {}
        for embeds, mask in pairs:
            if mask is None:
                raise EngineError("reference conditioning is missing its cached image_pad_mask")
            box[id(embeds)] = (embeds, mask.to(self._device))

        def encode_prompt(*args, **kwargs):
            embeds = kwargs.get("prompt_embeds")
            if embeds is not None and kwargs.get("image_pad_mask") is None:
                entry = box.get(id(embeds))
                if entry is not None and entry[0] is embeds:
                    kwargs["image_pad_mask"] = entry[1]
            return original(*args, **kwargs)

        pipe.encode_prompt = encode_prompt

        def remove():
            if "encode_prompt" in vars(pipe):
                del pipe.encode_prompt
            box.clear()
        return remove

    def _finite(self, tensor, shape, dtype):
        torch = self._torch
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != dtype or tensor.device != self._device
                or tensor.layout != torch.strided
                or not bool(torch.isfinite(tensor).all())):
            raise EngineError("component returned invalid shape/dtype/device or nonfinite values")

    def denoise(self, config, conditioning):
        self._check("encoded", config)
        torch = self._torch
        if conditioning is not self._conditioning:
            raise EngineError("conditioning is not owned by this adapter")
        # Repeat strict value validation at the caller-return boundary: a
        # mutable tensor must not be swapped/changed between encode and sample.
        from qwen_engine.encoder_production import validate_branch_bundle, validate_production_request
        # Never replace the retained child's runtime identity with parent facts.
        validate_branch_bundle(validate_production_request(self._encoding_request), conditioning)
        if conditioning.positive.embeds.shape[-1] != self._pipeline.transformer.config.context_in_dim:
            raise EngineError("conditioning hidden width disagrees with the transformer")
        kwargs = {}
        for prefix, branch in (("", conditioning.positive), ("negative_", conditioning.negative)):
            if branch is None:
                kwargs[prefix + "prompt_embeds"] = None
                kwargs[prefix + "prompt_embeds_mask"] = None
                continue
            embeds = branch.embeds.to(device=self._device, dtype=self._dtype)
            self._finite(embeds, tuple(branch.embeds.shape), self._dtype)
            kwargs[prefix + "prompt_embeds"] = embeds
            kwargs[prefix + "prompt_embeds_mask"] = (
                None if branch.attention_mask is None else branch.attention_mask.to(self._device))
        references = tuple(getattr(config, "references", ()))
        images = self.reference_images(config) if references else None
        mask_hook = None
        if images is not None:
            mask_hook = self._install_image_pad_mask_hook(
                [(kwargs["prompt_embeds"], conditioning.positive.image_pad_mask)]
                + ([] if conditioning.negative is None else
                   [(kwargs["negative_prompt_embeds"], conditioning.negative.image_pad_mask)]))
        generator = torch.Generator(device=self._device)
        self.actual_seed = generator.seed() if config.seed is None else config.seed
        # Domain separation: an edit run seeds the SAMPLER with a value
        # derived from (seed, ordered reference ids) so its initial noise can
        # never be the noise that produced one of its own references
        # (FINDINGS-edit-grid.md root cause). T2I keeps config.seed exactly.
        noise_seed = (derive_edit_noise_seed(self.actual_seed, references)
                      if references else self.actual_seed)
        self._noise_seed = noise_seed if references else None
        generator.manual_seed(noise_seed)
        if current() is not None:
            current().metadata["seed"] = self.actual_seed
            if references:
                current().metadata["noise_seed"] = noise_seed
                current().metadata["noise_seed_derivation"] = EDIT_NOISE_DERIVATION

        steps_observed: list[int] = []

        def callback(pipe, index, timestep, tensors):
            steps_observed.append(index)
            self._check("encoded", config)
            self._finite(tensors["latents"], (1, config.latent_tokens, 64), self._dtype)
            if self._progress_callback is not None:
                self._progress_callback(index + 1, config.steps)
            return {}

        try:
            with torch.inference_mode():
                result = self._pipeline(prompt=None, image=images, height=config.height, width=config.width,
                    num_inference_steps=config.steps, true_cfg_scale=config.true_cfg_scale if config.cfg_active else 1,
                    num_images_per_prompt=1, generator=generator, output_type="latent",
                    use_kv_cache=self.use_kv_cache, callback_on_step_end=callback, **kwargs)
        finally:
            if mask_hook is not None:
                mask_hook()
        self._check("encoded", config)
        self._finite(result.images, (1, config.latent_tokens, 64), self._dtype)
        if config.recipe is Recipe.TURBO and len(steps_observed) != TURBO_STEPS:
            # The saved sigmas make num_inference_steps inert inside the
            # pipeline; without this witness a tampered/truncated schedule
            # would run a different number of denoising steps than the
            # recipe claims (turbo-plan §3.1).
            raise EngineError(
                f"the turbo recipe must consume exactly the {TURBO_STEPS} "
                f"shipped sigmas; the sampler reported "
                f"{len(steps_observed)} step callbacks")
        self._state = "denoised"
        return result.images

    def decode(self, latents):
        self._check("denoised")
        torch, pipe, config = self._torch, self._pipeline, self._config
        self._finite(latents, (1, config.latent_tokens, 64), self._dtype)
        with torch.inference_mode():
            unpacked = pipe._unpack_latents(latents, config.height, config.width, pipe.vae_scale_factor)
            # Validate against the VAE's own dtype: memory-first tiled decode
            # runs the VAE in fp32 (the fp16 tiled path is non-finite at 2K),
            # while the execution tensors stay fp16.
            vae_dtype = pipe.vae.dtype
            mean = torch.tensor(pipe.vae.config.latents_mean, device=self._device, dtype=vae_dtype).view(1, 64, 1, 1, 1)
            std = torch.tensor(pipe.vae.config.latents_std, device=self._device, dtype=vae_dtype).view(1, 64, 1, 1, 1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
                raise EngineError("invalid VAE latent normalization")
            normalized = unpacked.to(vae_dtype) * std + mean
            self._finite(normalized, tuple(unpacked.shape), vae_dtype)
            with stage("vae_decoding", cuda=True):
                decoded = pipe.vae.decode(normalized, return_dict=False)[0][:, :, 0]
                self._finite(decoded, (1, 4, config.height, config.width), vae_dtype)
            with stage("output_conversion", cuda=True):
                images = pipe.image_processor.postprocess(decoded, output_type="pil")
        self._check("denoised")
        if len(images) != 1 or not isinstance(images[0], self._image.Image) or images[0].size != (config.width, config.height):
            raise EngineError("VAE postprocessor did not return one correctly sized image")
        self._state = "decoded"
        return images[0]

    def save(self, image, config):
        self._check("decoded", config)
        if (config.output_dir is None or not isinstance(image, self._image.Image)
                or image.mode not in ("RGB", "RGBA") or image.size != (config.width, config.height)):
            raise EngineError("an explicit output directory and correctly sized RGB/RGBA image are required")
        from qwen_engine.image_output import save_output_image
        path = save_output_image(image, config.output_dir, config.output_format.value,
                                 self.actual_seed,
                                 check_cancelled=lambda: self._check("decoded", config))
        self._state = "saved"
        return path

    def close(self):
        # Release only owned references. No global allocator flush, no host
        # changes, no deletion of saved artifacts, no forced thread termination.
        self._pipeline = self._encoder = self._manifest = self._bundle = None
        self._torch = self._image = self._device = self._dtype = self._config = None
        self._encoding_request = self._event = None
        self._progress_callback = None
        self._conditioning = None
        self._reference_pils = None
        self._noise_seed = None
        self._state = "closed"
