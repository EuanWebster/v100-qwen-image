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


def test_loader_records_state_dict_auxiliaries_and_placement_stages(monkeypatch):
    """The 65.5 s Q8 model_initialisation investigation added transparent
    stage probes; a real run must be able to tell GGUF state-dict loading
    apart from CUDA placement. Fakes keep this ML-free (diffusers/gguf are
    never imported by the guard)."""
    import sys
    import types
    from qwen_engine.adapters.q8_loader import load_q8_transformer
    from qwen_engine.performance import PerformanceRecorder

    gguf = types.ModuleType("gguf")
    gguf.GGMLQuantizationType = SimpleNamespace(Q8_0="Q8_0", F16="F16",
                                                F32="F32", BF16="BF16")
    gguf.GGUFReader = lambda path: SimpleNamespace(
        tensors=[SimpleNamespace(tensor_type="Q8_0")])
    diffusers = types.ModuleType("diffusers")
    diffusers.GGUFQuantizationConfig = lambda **kw: "qcfg"
    utils = types.ModuleType("diffusers.quantizers.gguf.utils")
    utils.ops = None
    utils.GGUFParameter = type("GGUFParameter", (), {})
    utils.dequantize_gguf_tensor = lambda p: p
    for name, module in (("gguf", gguf), ("diffusers", diffusers),
                         ("diffusers.quantizers", types.ModuleType("diffusers.quantizers")),
                         ("diffusers.quantizers.gguf", types.ModuleType("diffusers.quantizers.gguf")),
                         ("diffusers.quantizers.gguf.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)

    events = []
    model = SimpleNamespace(training=True,
                            named_parameters=lambda: [],
                            get_submodule=lambda name: None,
                            to=lambda device=None: events.append(("to", device)),
                            eval=lambda: events.append("eval"))

    def from_single_file(path, **kwargs):
        events.append("state_dict")
        assert kwargs["local_files_only"] is True
        return model

    cls = SimpleNamespace(from_single_file=staticmethod(from_single_file))
    bundle = SimpleNamespace(root="/models/assets")
    artifact = SimpleNamespace(path="/data/q8.gguf")
    recorder = PerformanceRecorder()
    with recorder.activate():
        returned = load_q8_transformer(cls, bundle, artifact,
                                       SimpleNamespace(float16="fp16"), "cuda:0")
    assert returned is model and events[-1] == "eval"  # fake eval never flips training
    assert [record["stage"] for record in recorder.records] == [
        "q8_gguf_state_dict", "q8_auxiliaries", "q8_device_placement"]
    assert events == ["state_dict", ("to", "cuda:0"), "eval"]
