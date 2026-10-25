"""Single-use phase bridge over explicitly supplied, trusted components.

No loader, resource approval, checkpoint provenance or public backend selector.
The caller owns those gates BEFORE transferring resident components here.
Plain import is ML-free; the default CLI factory remains absent/disabled.
"""
from __future__ import annotations

import io
import threading
import uuid
from pathlib import Path

from qwen_engine.config import AttentionBackend, ConfigError, GenerationConfig, OffloadMode, Quant
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
        components = (pipe, pipe.transformer, pipe.vae, pipe.scheduler)
        if any(type(value) is not cls for value, cls in zip(components, classes)):
            raise EngineError("supplied components must be the exact pinned Qwen21 classes")
        if pipe.text_encoder is not None or getattr(pipe, "_all_hooks", None):
            raise EngineError("resident generation must not retain an encoder/offload hooks")
        dtype = {"fp16": torch.float16, "fp32": torch.float32}[self.execution_dtype]
        device = pipe._execution_device
        if self.execution_dtype == "fp32" and str(device).split(":", 1)[0] != "cpu":
            raise EngineError("fp32 supplied execution is CPU-witness-only")
        for model in (pipe.transformer, pipe.vae):
            if any(getattr(module, "_hf_hook", None) is not None for module in model.modules()):
                raise EngineError("resident components must not have dispatch/offload hooks")
            if model is pipe.transformer and config.quant is Quant.Q8:
                from qwen_engine.adapters.q8_loader import validate_q8_parameters
                validate_q8_parameters(model, device, dtype)
            elif any(p.is_meta or p.device != device or p.dtype != dtype for p in model.parameters()):
                raise EngineError("supplied model placement/dtype disagrees with execution policy")
            if model.training:
                raise EngineError("supplied components must already be in evaluation mode")
        if (pipe.vae_scale_factor != 16 or pipe.latent_channels != 64
                or pipe.vae.config.z_dim != 64 or pipe.transformer.config.in_channels != 64
                or pipe.transformer.config.out_channels != 64):
            raise EngineError("unsupported latent architecture")
        pipe.transformer.set_attention_backend("native")
        if config.vae_tiled_decode:
            pipe.vae.enable_tiling()
        else:
            pipe.vae.disable_tiling()
        self._torch, self._image, self._device, self._dtype = torch, image, device, dtype
        self._config, self._state = config, "loaded"

    def encode(self, config):
        self._check("loaded", config)
        try:
            encoding = self._encoder.encode(config, encoder_dtype=self.encoder_dtype, cancel_event=self._event)
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
        }

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
        generator = torch.Generator(device=self._device)
        self.actual_seed = generator.seed() if config.seed is None else config.seed
        generator.manual_seed(self.actual_seed)
        if current() is not None:
            current().metadata["seed"] = self.actual_seed

        def callback(pipe, index, timestep, tensors):
            self._check("encoded", config)
            self._finite(tensors["latents"], (1, config.latent_tokens, 64), self._dtype)
            if self._progress_callback is not None:
                self._progress_callback(index + 1, config.steps)
            return {}

        with torch.inference_mode():
            result = self._pipeline(prompt=None, image=None, height=config.height, width=config.width,
                num_inference_steps=config.steps, true_cfg_scale=config.true_cfg_scale if config.cfg_active else 1,
                num_images_per_prompt=1, generator=generator, output_type="latent",
                use_kv_cache=self.use_kv_cache, callback_on_step_end=callback, **kwargs)
        self._check("encoded", config)
        self._finite(result.images, (1, config.latent_tokens, 64), self._dtype)
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
        self._state = "closed"
