#!/usr/bin/env python3
"""The GPU kernels' linear-algebra ops, lowered for the CPU instead and run against a reference.

Each kernel's MLIR source holds one linalg op, which the GPU build tiles, maps and lowers to
SPIR-V. This script takes the same source, lowers the op straight to loops and LLVM through
MLIR's own CPU path, runs it in MLIR's JIT, and compares every output bit for bit against a
reference computed without MLIR. It checks what the op means, the formula, the layout it reads
and the order and precision of its arithmetic, not what the GPU schedule does with it, which
needs the device.

    gemm     the GEMM template instantiated at a small shape, against numpy accumulating in the
             same order and precision: sequential along K, each product and sum rounded to the
             accumulator type.

The check also requires accumulating in the other type to give other bits, so that it would
fail loudly rather than pass if it stopped discriminating. Run by the gpu-kernels package; the
MLIR Python bindings must be on PYTHONPATH and the MLIR library directory given with --mlir-lib.
"""

import argparse
import ctypes
import os
import sys

import numpy as np
from mlir import ir, passmanager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor

from build import instantiate

CPU_PIPELINE = (
    "builtin.module("
    "one-shot-bufferize,"
    "convert-linalg-to-loops,"
    "convert-scf-to-cf,"
    "convert-to-llvm,"
    "reconcile-unrealized-casts)"
)


def run_cpu(source, entry, arrays, mlir_lib):
    """Lowers `source` for the CPU and calls `entry` on `arrays` (all memref arguments).

    The arrays are passed by descriptor and written in place, as the kernel's output
    argument is on the GPU.
    """
    with ir.Context():
        module = ir.Module.parse(source)
        for op in module.body.operations:
            if op.operation.name == "func.func":
                op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        passmanager.PassManager.parse(CPU_PIPELINE).run(module.operation)
        engine = ExecutionEngine(
            module,
            opt_level=2,
            shared_libs=[os.path.join(mlir_lib, "libmlir_c_runner_utils.so")],
        )
        args = [
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a)))
            for a in arrays
        ]
        engine.invoke(entry, *args)


def mismatches(got, want):
    """Indices where two arrays of the same dtype differ in their bits."""
    width = {2: np.uint16, 4: np.uint32}[got.dtype.itemsize]
    return np.flatnonzero(got.view(width).ravel() != want.view(width).ravel())


def report(name, got, want):
    bad = mismatches(got, want)
    flat_got, flat_want = got.ravel(), want.ravel()
    if bad.size:
        first = [(int(i), float(flat_got[i]), float(flat_want[i])) for i in bad[:4]]
        print(f"{name}: {bad.size} of {got.size} elements differ; first {first}")
    else:
        print(f"{name}: all {got.size} elements bit-identical to the reference")
    return bad.size


def reference_gemm(a, b, acc_type):
    """C = A * B^T, accumulated sequentially along K in `acc_type`, as the loops MLIR's CPU
    lowering emits accumulate: each product rounded to `acc_type`, then each sum."""
    m, k = a.shape
    n = b.shape[0]
    acc = np.zeros((m, n), dtype=acc_type)
    a_t, b_t = a.astype(acc_type), b.astype(acc_type)
    for kk in range(k):
        prod = (a_t[:, kk, None] * b_t[None, :, kk]).astype(acc_type)
        acc = (acc + prod).astype(acc_type)
    return acc


def check_gemm(args):
    template = open(args.source).read()
    m, n, k = args.m, args.n, args.k
    source = instantiate(template, M=m, N=n, K=k, ACC=args.acc)
    acc_type = {"f16": np.float16, "f32": np.float32}[args.acc]
    rng = np.random.default_rng(20261012)
    # Values of a typical activation and weight magnitude, rounded to f16; scaled so a
    # K-long f16 sum stays far from overflow.
    a = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)
    b = (rng.standard_normal((n, k)) * 0.05).astype(np.float16)
    want = reference_gemm(a, b, acc_type)
    out = np.zeros((m, n), dtype=acc_type)
    run_cpu(source, args.entry, [a, b, out], args.mlir_lib)
    failed = report(f"gemm {args.acc} accumulation {m}x{n}x{k}", out, want)
    # The check discriminates only if accumulating in the other type would have given other
    # bits; how far apart the two are is the scale of f16 accumulation's error at this K.
    other_type = np.float32 if acc_type == np.float16 else np.float16
    other = reference_gemm(a, b, other_type).astype(acc_type)
    differ = mismatches(other, want).size
    if not differ:
        sys.exit(
            "gemm self-test: f16 and f32 accumulation agree here; the case cannot tell them apart"
        )
    gap = np.abs(other.astype(np.float64) - want.astype(np.float64))
    scale = np.abs(want.astype(np.float64)).max()
    print(
        f"gemm self-test: {other_type.__name__} accumulation would differ in {differ} of {want.size} "
        f"elements, by at most {gap.max():.3g} against outputs up to {scale:.3g}"
    )
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mlir-lib", required=True)
    sub = parser.add_subparsers(dest="kernel", required=True)
    g = sub.add_parser("gemm")
    g.add_argument("source")
    g.add_argument("--entry", required=True)
    g.add_argument("--acc", choices=["f16", "f32"], required=True)
    g.add_argument("-m", type=int, required=True)
    g.add_argument("-n", type=int, required=True)
    g.add_argument("-k", type=int, required=True)
    args = parser.parse_args()
    sys.exit(1 if check_gemm(args) else 0)


if __name__ == "__main__":
    main()
