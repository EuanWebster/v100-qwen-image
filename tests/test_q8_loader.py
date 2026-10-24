"""Regression for packed BF16 auxiliaries outside GGUFLinear (ML-free)."""
from types import SimpleNamespace

from qwen_engine.adapters.q8_loader import _restore_bf16_auxiliaries, artifact_set_identity


def test_bf16_norm_auxiliary_is_restored_without_expanding_q8_weights():
    class Packed:
        def __init__(self, kind): self.quant_type = kind
    class FloatWeight:
        def to(self, dtype):
            assert dtype == 'fp16'
            return self
    packed_norm, packed_linear = Packed('BF16'), Packed('Q8_0')
    plain = object()
    owners = {'norm': SimpleNamespace(_parameters={'weight': packed_norm}),
              'linear': SimpleNamespace(_parameters={'weight': packed_linear}),
              'bias': SimpleNamespace(_parameters={'weight': plain})}
    model = SimpleNamespace(
        named_parameters=lambda: [(name+'.weight', owner._parameters['weight']) for name, owner in owners.items()],
        get_submodule=owners.__getitem__)
    unpacked = FloatWeight()
    seen = []
    def dequantize(value):
        seen.append(value)
        return unpacked
    def parameter(value, *, requires_grad):
        assert requires_grad is False
        return value
    torch = SimpleNamespace(float16='fp16', nn=SimpleNamespace(Parameter=parameter))
    utils = SimpleNamespace(GGUFParameter=Packed, dequantize_gguf_tensor=dequantize)
    _restore_bf16_auxiliaries(model, torch, utils, 'BF16')
    assert seen == [packed_norm]
    assert owners['norm']._parameters['weight'] is unpacked
    assert owners['linear']._parameters['weight'] is packed_linear
    assert owners['bias']._parameters['weight'] is plain


def test_quantized_artifact_changes_invalidate_combined_identity():
    assert artifact_set_identity('a'*64, 'b'*64) != artifact_set_identity('a'*64, 'c'*64)
    assert artifact_set_identity('a'*64, 'b'*64) != artifact_set_identity('d'*64, 'b'*64)
