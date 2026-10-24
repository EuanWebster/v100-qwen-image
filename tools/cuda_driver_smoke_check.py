#!/usr/bin/env python3
"""Explicit, small V100 CUDA check using only Python's stdlib and libcuda.

Run in a fresh process with an external timeout. JITs sm_70 PTX, transfers
4096-element inputs and checks every FP32/FP16 addition result on the CPU.
No clocks, ECC settings, services or model weights are changed. This is not
PyTorch, tensor-core, Qwen numerical, bandwidth or sustained-load acceptance.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import re
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pci-bus-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", args.pci_bus_id):
        parser.error("expected PCI address dddd:bb:dd.f")

    cuda = C.CDLL("libcuda.so.1")
    ptr, u64, uint, integer = C.c_void_p, C.c_uint64, C.c_uint, C.c_int
    signatures = {
        "cuInit": [uint],
        "cuGetErrorName": [integer, C.POINTER(C.c_char_p)],
        "cuDeviceGetByPCIBusId": [C.POINTER(integer), C.c_char_p],
        "cuDeviceGetName": [ptr, integer, integer],
        "cuDeviceComputeCapability": [C.POINTER(integer), C.POINTER(integer), integer],
        "cuCtxCreate_v2": [C.POINTER(ptr), uint, integer],
        "cuCtxDestroy_v2": [ptr],
        "cuModuleLoadData": [C.POINTER(ptr), ptr],
        "cuModuleGetFunction": [C.POINTER(ptr), ptr, C.c_char_p],
        "cuMemAlloc_v2": [C.POINTER(u64), C.c_size_t],
        "cuMemcpyHtoD_v2": [u64, ptr, C.c_size_t],
        "cuMemcpyDtoH_v2": [ptr, u64, C.c_size_t],
        "cuLaunchKernel": [ptr, uint, uint, uint, uint, uint, uint, uint, ptr, C.POINTER(ptr), C.POINTER(ptr)],
        "cuCtxSynchronize": [],
    }
    for name, types in signatures.items():
        function = getattr(cuda, name)
        function.argtypes, function.restype = types, integer

    def call(name, *values):
        status = getattr(cuda, name)(*values)
        if status:
            description = C.c_char_p()
            cuda.cuGetErrorName(status, C.byref(description))
            message = description.value.decode("ascii") if description.value else str(status)
            raise RuntimeError(f"{name}: {message}")

    call("cuInit", 0)
    device, major, minor = integer(), integer(), integer()
    call("cuDeviceGetByPCIBusId", C.byref(device), args.pci_bus_id.encode("ascii"))
    name = C.create_string_buffer(256)
    call("cuDeviceGetName", name, len(name), device)
    call("cuDeviceComputeCapability", C.byref(major), C.byref(minor), device)
    if "V100" not in name.value.decode("ascii") or (major.value, minor.value) != (7, 0):
        raise RuntimeError("selected device is not a Volta V100")

    count = 4096
    left = [(i % 97) / 4 - 12 for i in range(count)]
    right = [(i % 31) / 2 + 1 for i in range(count)]
    expected = [a + b for a, b in zip(left, right)]
    context = ptr()
    call("cuCtxCreate_v2", C.byref(context), 0, device)
    results = []
    try:
        for precision, pack_type, register_type, load_type, item_bytes in (
            ("f32", "f", "f32", "f32", 4),
            ("f16", "e", "b16", "b16", 2),
        ):
            ptx = f""".version 6.0
.target sm_70
.address_size 64
.visible .entry vector_add(.param .u64 a, .param .u64 b,
                          .param .u64 out, .param .u32 n) {{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;
    .reg .{register_type} %v<3>;
    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [b];
    ld.param.u64 %rd2, [out];
    ld.param.u32 %r4, [n];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    setp.ge.u32 %p, %r3, %r4;
    @%p bra done;
    mul.wide.u32 %rd3, %r3, {item_bytes};
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;
    ld.global.{load_type} %v0, [%rd4];
    ld.global.{load_type} %v1, [%rd5];
    add.rn.{precision} %v2, %v0, %v1;
    st.global.{load_type} [%rd6], %v2;
done:
    ret;
}}
"""
            image = C.create_string_buffer(ptx.encode("ascii"))
            module, kernel = ptr(), ptr()
            call("cuModuleLoadData", C.byref(module), image)
            call("cuModuleGetFunction", C.byref(kernel), module, b"vector_add")
            byte_count = count * item_bytes
            buffers = [u64(), u64(), u64()]
            for buffer in buffers:
                call("cuMemAlloc_v2", C.byref(buffer), byte_count)
            for buffer, values in zip(buffers, (left, right)):
                host = C.create_string_buffer(struct.pack(f"<{count}{pack_type}", *values))
                call("cuMemcpyHtoD_v2", buffer, host, byte_count)
            # Poison output so an unwritten result cannot pass the CPU check.
            poison = C.create_string_buffer(b"\xff" * byte_count)
            call("cuMemcpyHtoD_v2", buffers[2], poison, byte_count)
            length = uint(count)
            parameters = (ptr * 4)(*[C.cast(C.pointer(value), ptr) for value in (*buffers, length)])
            call("cuLaunchKernel", kernel, (count + 127) // 128, 1, 1, 128, 1, 1, 0, None, parameters, None)
            call("cuCtxSynchronize")
            output = C.create_string_buffer(byte_count)
            call("cuMemcpyDtoH_v2", output, buffers[2], byte_count)
            if output.raw != struct.pack(f"<{count}{pack_type}", *expected):
                raise RuntimeError(f"{precision}: GPU output differs from CPU expectation")
            results.append({"precision": precision, "elements": count, "result": "PASS"})
    finally:
        # A private context owns all allocations/modules, including on failure.
        call("cuCtxDestroy_v2", context)

    print(json.dumps({"device": name.value.decode("ascii"), "pci_bus_id": args.pci_bus_id,
                      "compute_capability": [major.value, minor.value], "checks": results,
                      "context_destroyed": True}, indent=2))


if __name__ == "__main__":
    main()
