"""Owned lifecycle over explicit CPU fixture adapters, no ML/real images."""
from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path

import pytest

from qwen_engine.adapters.base import GenerationAdapter, ManagedGenerationAdapter
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.core import generate
from qwen_engine.errors import GenerationCancelled
from qwen_engine.jobs import JobController, JobState


class Adapter:
    def __init__(self, *, fail=None, close_error=None, cancel_at=None):
        self.calls = []
        self.fail, self.close_error, self.cancel_at = fail, close_error, cancel_at
        self.event = None

    def set_cancel_event(self, event):
        self.calls.append("bind")
        self.event = event
        self.step("bind")

    def step(self, name):
        if name == self.cancel_at:
            self.event.set()
        if name == self.fail:
            raise RuntimeError("fixture primary failure")

    def load(self, config, assets):
        self.calls.append("load")
        self.step("load")

    def encode(self, config):
        self.calls.append("encode")
        self.step("encode")
        return object()

    def denoise(self, config, conditioning):
        self.calls.append("denoise")
        self.step("denoise")
        return object()

    def decode(self, latents):
        self.calls.append("decode")
        self.step("decode")
        return object()

    def save(self, image, config):
        self.calls.append("save")
        self.step("save")
        path = Path(config.output_dir) / "fixture.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture result not a real image")
        return path

    def close(self):
        self.calls.append("close")
        if self.close_error is not None:
            raise self.close_error


def test_managed_protocol_and_order(tmp_path):
    adapter = Adapter()
    assert isinstance(adapter, GenerationAdapter) and isinstance(adapter, ManagedGenerationAdapter)
    reported = []
    result = generate(GenerationConfig(output_dir=tmp_path), adapter, {}, phase_callback=reported.append)
    assert adapter.calls == ["bind", "load", "encode", "denoise", "decode", "save", "close"]
    assert reported == ["load", "encode", "denoise", "decode", "save"]
    assert Path(result.output_path).is_file()


@pytest.mark.parametrize("phase", ["bind", "load", "encode", "denoise", "decode", "save"])
def test_partial_failure_always_closes_and_preserves_primary(tmp_path, phase):
    adapter = Adapter(fail=phase, close_error=OSError("fixture cleanup failure"))
    with pytest.raises(RuntimeError, match="primary failure"):
        generate(GenerationConfig(output_dir=tmp_path), adapter, {})
    assert adapter.calls[-1] == "close" and adapter.calls.count("close") == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("phase", ["bind", "load", "encode", "denoise", "decode"])
def test_cancellation_stops_next_phase_and_closes(tmp_path, phase):
    event = threading.Event()
    adapter = Adapter(cancel_at=phase)
    with pytest.raises(GenerationCancelled):
        generate(GenerationConfig(output_dir=tmp_path), adapter, {}, cancel_event=event)
    assert "save" not in adapter.calls and adapter.calls[-1] == "close"


def test_pre_cancelled_run_never_loads(tmp_path):
    event = threading.Event()
    event.set()
    adapter = Adapter()
    with pytest.raises(GenerationCancelled):
        generate(GenerationConfig(output_dir=tmp_path), adapter, {}, cancel_event=event)
    assert adapter.calls == ["bind", "close"]


def test_cancel_request_during_verified_save_does_not_hide_result(tmp_path):
    adapter = Adapter(cancel_at="save", close_error=OSError("fixture cleanup fault"))
    result = generate(GenerationConfig(output_dir=tmp_path), adapter, {}, cancel_event=threading.Event())
    assert Path(result.output_path).is_file() and adapter.calls[-1] == "close"


def test_callback_can_cancel_before_phase_body(tmp_path):
    event = threading.Event()
    adapter = Adapter()
    def cancel(phase):
        if phase == "denoise":
            event.set()
    with pytest.raises(GenerationCancelled):
        generate(GenerationConfig(output_dir=tmp_path), adapter, {}, cancel_event=event, phase_callback=cancel)
    assert adapter.calls == ["bind", "load", "encode", "close"]


def test_callback_fault_and_interrupt_preserve_identity(tmp_path):
    for error in (ValueError("callback fault"), KeyboardInterrupt()):
        adapter = Adapter(close_error=OSError("cleanup fault"))
        def callback(phase):
            raise error
        with pytest.raises(type(error)) as caught:
            generate(GenerationConfig(output_dir=tmp_path), adapter, {}, phase_callback=callback)
        assert caught.value is error and adapter.calls == ["bind", "close"]


@pytest.mark.parametrize("option,value", [("cancel_event", True), ("cancel_event", object()),
    ("phase_callback", 1), ("phase_callback", [])])
def test_invalid_options_before_adapter_ownership(tmp_path, option, value):
    adapter = Adapter()
    with pytest.raises(ConfigError):
        generate(GenerationConfig(output_dir=tmp_path), adapter, {}, **{option: value})
    assert not adapter.calls


def test_unconfigured_output_never_enters_adapter(tmp_path):
    adapter = Adapter()
    with pytest.raises(ConfigError):
        generate(GenerationConfig(), adapter, {})
    assert not adapter.calls


def test_job_controller_recognizes_engine_cancellation(tmp_path):
    entered, release = threading.Event(), threading.Event()
    adapter = Adapter()
    def execute(context):
        entered.set()
        assert release.wait(3)
        return generate(context.config, adapter, {}, cancel_event=context._cancel, phase_callback=context.report_phase)
    controller = JobController(tmp_path, execute)
    try:
        record = controller.submit(GenerationConfig())
        assert entered.wait(2)
        controller.cancel(record.job_id)
        release.set()
        assert controller.wait(record.job_id).state == JobState.CANCELLED
        assert adapter.calls == ["bind", "close"]
    finally:
        release.set()
        assert controller.close()


class TensorOwner:
    """Weakref-able stand-in for an object owning CUDA tensors."""


@pytest.mark.parametrize("kind", ["cancelled", "failed", "gpu-oom"])
def test_terminal_job_releases_executor_frames_without_gc(tmp_path, kind):
    # Regression for the cancelled-run 14 GB retention: the adapter fails
    # inside the denoise frame while a tensor owner is still a live local.
    # If the engine kept the raised exception reachable through a reference
    # cycle (exception -> traceback -> frame -> frame local -> exception),
    # the owner could only die in a cyclic GC pass.  With gc disabled for
    # the whole observation window, refcounting alone must free the owner
    # by the time the real controller has recorded the terminal state.
    refs = []

    class HoldingAdapter(Adapter):
        def denoise(self, config, conditioning):
            owner = TensorOwner()
            refs.append(weakref.ref(owner))
            self.calls.append("denoise")
            if kind == "cancelled":
                self.event.set()
                raise GenerationCancelled("fixture mid-denoise cancellation")
            if kind == "gpu-oom":
                # torch-identity fake: jobs.py classifies CUDA OOM without
                # importing torch, and the classifier must NOT retain the
                # traceback (the denoise frame owns the tensor owner here)
                torch_oom = type("OutOfMemoryError", (RuntimeError,), {})
                torch_oom.__module__ = "torch"
                raise torch_oom("fixture mid-denoise CUDA out of memory")
            raise RuntimeError("fixture mid-denoise failure")

    adapter = HoldingAdapter()
    controller = JobController(
        tmp_path, lambda context: generate(
            context.config, adapter, {}, cancel_event=context._cancel,
            phase_callback=context.report_phase),
        # no-op boundary release: only refcounting may free the owner here
        memory_release=lambda: None)
    try:
        gc.collect()  # clear unrelated pre-existing cycles first
        gc.disable()
        record = controller.submit(GenerationConfig())
        done = controller.wait(record.job_id)
        assert refs and all(ref() is None for ref in refs)
    finally:
        gc.enable()
    assert done.state == (JobState.CANCELLED if kind == "cancelled" else JobState.FAILED)
    assert done.error_code == {"cancelled": None, "failed": "execution_failed",
                               "gpu-oom": "out_of_gpu_memory"}[kind]
    assert adapter.calls == ["bind", "load", "encode", "denoise", "close"]
    assert controller.close()
