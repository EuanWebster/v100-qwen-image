"""Reviewed Q8_0 GGUF transformer attachment; heavy imports stay inside calls.

The current pinned Diffusers implementation dequantizes each linear weight for
FP16 computation. No native INT8 matrix kernel, Triton, remote kernel download,
or hardware FP8 support is required by this path.
"""
from __future__ import annotations

import os
import hashlib
import json

from qwen_engine.errors import EngineError


def artifact_set_identity(base_identity: str, artifact_identity: str) -> str:
    """Bind base companions and the exact quantized transformer into evidence."""
    return hashlib.sha256(json.dumps({"schema": "qwen_engine.q8_bundle.v1",
                                     "base": base_identity, "q8": artifact_identity},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_q8_environment():
    """Refuse optional remote-kernel mode before any ML module can import it."""
    if os.environ.get("DIFFUSERS_GGUF_CUDA_KERNELS", "false").lower() in ("1", "true", "yes"):
        raise EngineError("Q8 profile requires DIFFUSERS_GGUF_CUDA_KERNELS=false")


def _restore_bf16_auxiliaries(model, torch, gguf_utils, bf16_type):
    for name, parameter in list(model.named_parameters()):
        if isinstance(parameter, gguf_utils.GGUFParameter) and parameter.quant_type == bf16_type:
            owner_name, _, parameter_name = name.rpartition(".")
            value = gguf_utils.dequantize_gguf_tensor(parameter).to(torch.float16)
            model.get_submodule(owner_name)._parameters[parameter_name] = torch.nn.Parameter(value, requires_grad=False)


def load_q8_transformer(transformer_cls, bundle, artifact, torch, device):
    # The optional kernel mode can execute code downloaded from the Hub. Keep
    # this profile on the pinned local pure-Torch fallback, refusing ambient
    # settings instead of mutating them or silently choosing another backend.
    validate_q8_environment()
    import gguf
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import utils as gguf_utils
    from qwen_engine.performance import stage

    if gguf_utils.ops is not None:
        raise EngineError("Q8 profile refuses previously initialized remote GGUF kernels")
    reader = gguf.GGUFReader(str(artifact.path))
    allowed = {gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.F16,
               gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16}
    types = {tensor.tensor_type for tensor in reader.tensors}
    if gguf.GGMLQuantizationType.Q8_0 not in types or not types <= allowed:
        raise EngineError("Q8 profile requires Q8_0 tensors with only unquantized auxiliary tensors")
    del reader
    # Upstream from_single_file -> load_gguf_checkpoint reads the GGUF through
    # a np.memmap and copies EVERY tensor into fresh heap
    # (torch.from_numpy(tensor.data.copy()), 265x ~16.8 s warm-measured here
    # for the 7.64 GB Q8_0 file; the copy is also why host RSS reaches
    # ~2x file size). Measured 2026-10-26; no engine-side hash happens in
    # this call and the artifact sha256 was verified exactly once per job
    # before it (resolve_q8_artifact). Staged below so the next real run
    # splits load time from device placement WITHOUT changing behaviour
    # (stage() is a no-op when no PerformanceRecorder is active). Wall-clock
    # only: they run inside the caller's CUDA-measured transformer_loading
    # stage, and CUDA performance scopes must not nest.
    with stage("q8_gguf_state_dict"):
        model = transformer_cls.from_single_file(
            str(artifact.path), config=str(bundle.root), subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
            torch_dtype=torch.float16, local_files_only=True,
        )
    # The upstream GGUF loader keeps BF16 auxiliaries as packed byte
    # GGUFParameters even in non-linear modules (notably text_norm). Those
    # modules do not dequantize in forward. Normalize only these unquantized
    # auxiliaries on CPU before placement; actual Q8 weights remain packed.
    with stage("q8_auxiliaries"):
        _restore_bf16_auxiliaries(model, torch, gguf_utils, gguf.GGMLQuantizationType.BF16)
    with stage("q8_device_placement"):
        model.to(device=device)
    model.eval()
    return model


def validate_q8_parameters(model, device, dtype):
    """Recognize only the pinned GGUF parameter class and reviewed Q8 type."""
    import gguf
    import torch
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter

    count = 0
    for name, parameter in model.named_parameters():
        if parameter.is_meta or parameter.device != device:
            raise EngineError("Q8 transformer contains misplaced parameters")
        if isinstance(parameter, GGUFParameter):
            owner = model.get_submodule(name.rpartition(".")[0])
            if (parameter.quant_type != gguf.GGMLQuantizationType.Q8_0
                    or parameter.dtype != torch.uint8 or not isinstance(owner, GGUFLinear)):
                raise EngineError("Q8 transformer contains an unreviewed quantized parameter")
            count += parameter.quant_type == gguf.GGMLQuantizationType.Q8_0
        elif parameter.dtype != dtype:
            raise EngineError("Q8 transformer auxiliary parameters disagree with FP16 execution")
    if count == 0:
        raise EngineError("Q8 transformer has no quantized parameters")
    return count
