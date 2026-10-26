#!/usr/bin/env python3
"""Compiles one GPU kernel from its MLIR source to SPIR-V, checks the result, and writes the
manifest a runtime loads it by.

    build.py --out DIR gemm --m 512 --n 6144 --k 2048 --acc f16

Each kernel is a linear-algebra op in `<kernel>.mlir` and a transform-dialect schedule in
`<kernel>.schedule.mlir` that tiles it and maps it onto workgroups and subgroups; `target.mlir`
describes the device. `pipeline()` and `SPIRV_MODULE_PIPELINE` below are the whole
compilation, in order. After them this takes the one spirv.module out of the result, makes
two corrections upstream MLIR does not make (`fix_module`), serializes it, validates it with
spirv-val, and reads the binary's own disassembly to write `<name>.json`: entry point,
workgroup size, dispatch grid, bindings, the subgroup size the kernel was written for, and
the capabilities and Vulkan features it needs. Every field is either read from the compiled
IR or checked against the binary, so the manifest cannot claim what the SPIR-V does not do.

mlir-opt, mlir-translate, spirv-val and spirv-dis are taken from PATH, and the MLIR Python
bindings from PYTHONPATH; the gpu-kernels Nix package provides all of them.
"""

import argparse
import json
import os
import re
import subprocess
import sys

from mlir import ir

HERE = os.path.dirname(os.path.abspath(__file__))

# A kernel's interface, as its source declares it: one entry per memref argument, in order,
# which is also the binding order. `access` is how the kernel uses the buffer.
KERNELS = {
    "gemm": {
        "bindings": [
            ("a", "read", "activations, M x K f16, row-major"),
            ("b", "read", "weights, N x K f16, row-major (one row per output feature)"),
            ("c", "write", "result, M x N, row-major, in the accumulator type"),
        ],
    },
}

# The GEMM schedule's tiles: 128 x 128 of C per workgroup, 32 of K per step.
GEMM_TILE_MN, GEMM_TILE_K = 128, 32


def pipeline(kernel):
    """The pass pipeline up to SPIR-V, as mlir-opt arguments, for a kernel whose transform
    library is `<kernel>.schedule.mlir` with entry points `@<kernel>_schedule` and
    `@<kernel>_abi`. It has to be one invocation: the transform libraries name SPIR-V
    attributes, and mlir-opt loads the SPIR-V dialect before the pipeline runs only if a
    pass in the pipeline declares it (convert-gpu-to-spirv does); loading it while the pass
    manager runs aborts."""
    libraries = ",".join(
        os.path.join(HERE, f) for f in ("target.mlir", f"{kernel}.schedule.mlir")
    )
    return [
        # Load the schedule and the target description; run the schedule: tile, fuse,
        # vectorize, bufferize, and map the loops onto a gpu.launch's blocks, subgroups and
        # invocations.
        f"--transform-preload-library=transform-library-paths={libraries}",
        f"--transform-interpreter=entry-point={kernel}_schedule",
        # Scalar ops the schedule left as tiny linalg ops become loops; subviews fold into
        # the accesses that use them, and affine index maps become arith.
        "--convert-linalg-to-loops",
        "--fold-memref-alias-ops",
        "--lower-affine",
        "--canonicalize",
        "--cse",
        # Constants and index arithmetic move into the launch so they are not passed in as
        # arguments (each would become a binding), then the launch becomes a gpu.func.
        "--gpu-launch-sink-index-computations",
        "--gpu-kernel-outlining",
        # vector.contract on 16x16x16 tiles and their transfers -> gpu.subgroup_mma_*. Run
        # after outlining so the zero accumulator is created inside the kernel.
        "--convert-vector-to-gpu",
        "--canonicalize",
        "--cse",
        # The SPIR-V entry-point ABI (workgroup size), then the target environment.
        f"--transform-interpreter=entry-point={kernel}_abi",
        "--transform-interpreter=entry-point=attach_spirv_target",
        # gpu, arith, memref, scf -> spirv; gpu.subgroup_mma_* -> spirv.KHR.CooperativeMatrix*.
        "--convert-gpu-to-spirv",
        "--canonicalize",
        "--cse",
    ]


# Run as a second mlir-opt invocation, on the spirv.module alone: kernel arguments become
# descriptor-bound global variables, and the module is given the minimal version,
# capabilities and extensions its ops need.
SPIRV_MODULE_PIPELINE = (
    "--pass-pipeline=builtin.module(spirv.module("
    "spirv-lower-abi-attrs,spirv-update-vce,canonicalize,cse))"
)


def run(argv, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, **kwargs)
    if result.returncode:
        sys.stderr.write(result.stdout + result.stderr)
        sys.exit(f"failed: {' '.join(argv[:2])} ...")
    return result.stdout


def instantiate(template, **values):
    """Substitutes `@NAME@` placeholders, and refuses to leave one unsubstituted."""
    for name, value in values.items():
        template = template.replace(f"@{name}@", str(value))
    left = re.findall(r"@[A-Z]+@", template)
    if left:
        sys.exit(f"placeholders left unsubstituted: {sorted(set(left))}")
    return template


def fix_module(spv):
    """The two corrections upstream MLIR (at the pinned LLVM) does not make for Vulkan.

    The SPIR-V version: spirv-update-vce deduces the minimum from the ops alone and ignores
    decorations, so a module whose index arithmetic carries NoSignedWrap/NoUnsignedWrap
    (SPIR-V 1.4) can come out declaring 1.0, which spirv-val rejects. The module declares the
    target's version, 1.6, the one Vulkan 1.3 and later accept.

    The memory model: convert-gpu-to-spirv always picks GLSL450 for a shader, but a shader
    that declares CooperativeMatrixKHR must declare VulkanMemoryModel (SPV_KHR_cooperative_matrix,
    enforced by spirv-val). Such a module gets the Vulkan memory model and the capability.

    A bandaid with a known proper fix: teach upstream's spirv::getMemoryModel to choose
    Vulkan when the target environment allows it, and spirv-update-vce to count decorations'
    and the memory model's requirements.
    """
    vce = str(spv.attributes["vce_triple"])
    m = re.fullmatch(r"#spirv\.vce<v(\d+\.\d+), \[([^\]]*)\], \[([^\]]*)\]>", vce)
    if not m:
        sys.exit(f"unexpected vce triple {vce}")
    caps = [c.strip() for c in m.group(2).split(",") if c.strip()]
    exts = [e.strip() for e in m.group(3).split(",") if e.strip()]
    if "CooperativeMatrixKHR" in caps:
        if "VulkanMemoryModel" not in caps:
            caps.insert(1, "VulkanMemoryModel")
        spv.attributes["memory_model"] = ir.Attribute.parse(
            "#spirv.memory_model<Vulkan>"
        )
    spv.attributes["vce_triple"] = ir.Attribute.parse(
        f"#spirv.vce<v1.6, [{', '.join(caps)}], [{', '.join(exts)}]>"
    )


def kernel_facts(module):
    """What the compiled IR says about the kernel: its gpu.func's attributes and argument
    types. The gpu.module stays beside the spirv.module after conversion, for the host side."""
    funcs = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    if inner.operation.name == "gpu.func":
                        funcs.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    if len(funcs) != 1:
        sys.exit(f"expected one gpu.func, found {len(funcs)}")
    func = funcs[0].operation
    attrs = func.attributes
    block_size = list(ir.DenseI32ArrayAttr(attrs["known_block_size"]))
    grid_size = list(ir.DenseI32ArrayAttr(attrs["known_grid_size"]))
    subgroup = None
    if "kernel.required_subgroup_size" in attrs:
        subgroup = ir.IntegerAttr(attrs["kernel.required_subgroup_size"]).value
    args = []
    for arg in func.regions[0].blocks[0].arguments:
        t = ir.MemRefType(arg.type)
        args.append({"shape": list(t.shape), "element": str(t.element_type)})
    return {
        "block_size": block_size,
        "grid_size": grid_size,
        "subgroup": subgroup,
        "args": args,
    }


class Disassembly:
    """The facts a manifest needs, read from spirv-dis output (friendly ids)."""

    def __init__(self, text):
        version = re.search(r"^; Version: (\d+\.\d+)$", text, re.MULTILINE)
        self.version = version.group(1) if version else None
        self.lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.startswith(";")
        ]
        self.defs = {}
        for ln in self.lines:
            m = re.match(r"(%\S+) = (\S+) ?(.*)", ln)
            if m:
                self.defs[m.group(1)] = (m.group(2), m.group(3).split())

    def ops(self, opcode):
        out = []
        for ln in self.lines:
            words = ln.split()
            if words[0] == opcode:
                out.append(words[1:])
            elif len(words) > 2 and words[1] == "=" and words[2] == opcode:
                out.append([words[0]] + words[3:])
        return out

    def constant(self, ident):
        opcode, operands = self.defs[ident]
        assert opcode == "OpConstant", ident
        return int(operands[1])

    def scalar(self, ident):
        opcode, operands = self.defs[ident]
        if opcode == "OpTypeFloat":
            return f"f{operands[0]}"
        if opcode == "OpTypeInt":
            return f"i{operands[0]}"
        sys.exit(f"not a scalar type: {ident}")

    def bindings(self):
        """(set, binding) -> (element type, element count) of each storage buffer."""
        decorations = {}
        for operands in self.ops("OpDecorate"):
            if operands[1] in ("DescriptorSet", "Binding"):
                decorations.setdefault(operands[0], {})[operands[1]] = int(operands[2])
        out = {}
        for var, deco in decorations.items():
            _, (ptr_type, storage) = self.defs[var]
            assert storage == "StorageBuffer", (var, storage)
            _, (_, struct) = self.defs[ptr_type]
            _, (array,) = self.defs[struct]
            opcode, (element, length) = self.defs[array]
            assert opcode == "OpTypeArray", array
            out[(deco["DescriptorSet"], deco["Binding"])] = (
                self.scalar(element),
                self.constant(length),
                self.access(var),
            )
        return out

    def access(self, var):
        """ "read", "write" or "read-write": whether any load or store goes through a pointer
        derived from `var` by access chains."""
        derived = {var}
        chains = [
            ops
            for opcode in ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain")
            for ops in self.ops(opcode)
        ]
        grew = True
        while grew:
            grew = False
            for ops in chains:
                if ops[2] in derived and ops[0] not in derived:
                    derived.add(ops[0])
                    grew = True
        loads = [
            ops[2]
            for opcode in ("OpLoad", "OpCooperativeMatrixLoadKHR")
            for ops in self.ops(opcode)
        ]
        stores = [
            ops[0]
            for opcode in ("OpStore", "OpCooperativeMatrixStoreKHR")
            for ops in self.ops(opcode)
        ]
        reads = any(p in derived for p in loads)
        writes = any(p in derived for p in stores)
        return {
            (True, False): "read",
            (False, True): "write",
            (True, True): "read-write",
        }.get((reads, writes), "unused")


# OpTypeCooperativeMatrixKHR's Use operand: MatrixAKHR, MatrixBKHR, MatrixAccumulatorKHR.
USES = {0: "a", 1: "b", 2: "acc"}


def inspect(disassembly, facts, kernel):
    """Checks the binary against the compiled IR and the kernel's declared interface, and
    returns the manifest fields read from the binary."""
    d = Disassembly(disassembly)
    problems = []

    (entry,) = d.ops("OpEntryPoint")
    model, _, name = entry[0], entry[1], entry[2].strip('"')
    if model != "GLCompute":
        problems.append(f"entry point model {model}")
    (local,) = [ops for ops in d.ops("OpExecutionMode") if ops[1] == "LocalSize"]
    local_size = [int(v) for v in local[2:5]]
    if local_size != facts["block_size"]:
        problems.append(
            f"LocalSize {local_size} but the IR's block size is {facts['block_size']}"
        )

    declared = KERNELS[kernel]["bindings"]
    bindings = d.bindings()
    if sorted(bindings) != [(0, i) for i in range(len(declared))]:
        problems.append(
            f"bindings {sorted(bindings)}, declared {len(declared)} in set 0"
        )
    manifest_bindings = []
    for i, ((bname, access, what), arg) in enumerate(zip(declared, facts["args"])):
        element, length, used = bindings.get((0, i), (None, None, None))
        if used != access:
            problems.append(
                f"binding {i} ({bname}) is declared {access} but the SPIR-V uses it {used}"
            )
        count = 1
        for s in arg["shape"]:
            count *= s
        if element != arg["element"] or length != count:
            problems.append(
                f"binding {i}: SPIR-V {element}[{length}], IR {arg['element']}{arg['shape']}"
            )
        manifest_bindings.append(
            {
                "set": 0,
                "binding": i,
                "name": bname,
                "access": access,
                "description": what,
                "element": arg["element"],
                "shape": arg["shape"],
                "bytes": count * int(re.sub(r"\D", "", arg["element"])) // 8,
            }
        )

    if [ops for ops in d.ops("OpVariable") if ops[-1] == "PushConstant"]:
        problems.append(
            "the module has push constants; the manifest format does not describe them yet"
        )
    if [ops for ops in d.ops("OpDecorate") if ops[1] == "SpecId"]:
        problems.append(
            "the module has specialization constants; the manifest does not describe them yet"
        )
    workgroup_vars = [ops for ops in d.ops("OpVariable") if ops[-1] == "Workgroup"]

    coop_types = {}
    for ident, operands in d.defs.items():
        if operands[0] == "OpTypeCooperativeMatrixKHR":
            component, scope, rows, cols, use = operands[1]
            coop_types[ident] = {
                "element": d.scalar(component),
                "scope": d.constant(scope),
                "rows": d.constant(rows),
                "cols": d.constant(cols),
                "use": USES[d.constant(use)],
            }
    muladds = d.ops("OpCooperativeMatrixMulAddKHR")
    shapes = set()
    for ops in muladds:
        result_type, a, b, c = ops[1], ops[2], ops[3], ops[4]
        types = {}
        for role, ident in (("a", a), ("b", b), ("acc", c)):
            types[role] = coop_types[
                d.defs[ident][1][0]
            ]  # an operand's first word is its type
        shapes.add(
            (
                types["a"]["element"],
                types["b"]["element"],
                types["acc"]["element"],
                coop_types[result_type]["element"],
                types["a"]["rows"],
                types["b"]["cols"],
                types["a"]["cols"],
                types["a"]["scope"],
            )
        )
    for ct in coop_types.values():
        if (ct["rows"], ct["cols"]) != (16, 16) or ct["scope"] != 3:  # 3 = Subgroup
            problems.append(
                f"cooperative matrix {ct} is not RADV's 16x16 at subgroup scope"
            )

    rounding = [ops for ops in d.ops("OpDecorate") if ops[1] == "FPRoundingMode"]
    if problems:
        sys.exit(
            "binary does not match its IR or declared interface:\n  "
            + "\n  ".join(problems)
        )
    capabilities = [ops[0] for ops in d.ops("OpCapability")]
    return {
        "spirv_version": d.version,
        "entry_point": name,
        "workgroup_size": local_size,
        "bindings": manifest_bindings,
        "workgroup_variables": len(workgroup_vars),
        "capabilities": capabilities,
        "extensions": [ops[0].strip('"') for ops in d.ops("OpExtension")],
        "memory_model": d.ops("OpMemoryModel")[0][1],
        "cooperative_matrix_muladds": len(muladds),
        "cooperative_matrix": [
            {
                "m": m,
                "n": n,
                "k": k,
                "a": a,
                "b": b,
                "c": c,
                "result": r,
                "scope": "subgroup",
            }
            for (a, b, c, r, m, n, k, _) in sorted(shapes)
        ],
        "fp_rounding_modes": sorted({ops[2] for ops in rounding}),
    }


# The Vulkan feature each SPIR-V capability needs enabled, by VkPhysicalDevice*Features member.
FEATURES = {
    "Float16": "shaderFloat16",
    "Int8": "shaderInt8",
    "Int16": "shaderInt16",
    "StorageBuffer8BitAccess": "storageBuffer8BitAccess",
    "StorageBuffer16BitAccess": "storageBuffer16BitAccess",
    "VulkanMemoryModel": "vulkanMemoryModel",
    "CooperativeMatrixKHR": "cooperativeMatrix",
}


def build(args):
    kernel = args.kernel
    template = open(os.path.join(HERE, f"{kernel}.mlir")).read()
    substitutions = {"M": args.m, "N": args.n, "K": args.k, "ACC": args.acc}
    name = f"gemm_{args.acc}acc_{args.m}x{args.n}x{args.k}"
    source = instantiate(template, **substitutions)
    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "ir")
    os.makedirs(work, exist_ok=True)
    src_path = os.path.join(work, f"{name}.mlir")
    open(src_path, "w").write(source)

    lowered = run(["mlir-opt", src_path] + pipeline(kernel))
    lowered = run(["mlir-opt", "-", SPIRV_MODULE_PIPELINE], input=lowered)
    with ir.Context():
        module = ir.Module.parse(lowered)
        facts = kernel_facts(module)
        spv_modules = [
            op for op in module.body.operations if op.operation.name == "spirv.module"
        ]
        if len(spv_modules) != 1:
            sys.exit(f"expected one spirv.module, found {len(spv_modules)}")
        spv = spv_modules[0].operation
        fix_module(spv)
        spirv_text = str(spv)
    spirv_mlir = os.path.join(work, f"{name}.spirv.mlir")
    open(spirv_mlir, "w").write(spirv_text + "\n")

    spv_path = os.path.join(args.out, f"{name}.spv")
    run(
        [
            "mlir-translate",
            "--no-implicit-module",
            "--serialize-spirv",
            spirv_mlir,
            "-o",
            spv_path,
        ]
    )
    run(["spirv-val", "--target-env", "vulkan1.3", spv_path])
    disassembly = run(["spirv-dis", spv_path])
    open(os.path.join(args.out, f"{name}.spvasm"), "w").write(disassembly)

    found = inspect(disassembly, facts, kernel)
    acc = args.acc
    want = [
        {
            "m": 16,
            "n": 16,
            "k": 16,
            "a": "f16",
            "b": "f16",
            "c": acc,
            "result": acc,
            "scope": "subgroup",
        }
    ]
    if found["cooperative_matrix"] != want or not found["cooperative_matrix_muladds"]:
        sys.exit(
            f"{name}: expected OpCooperativeMatrixMulAddKHR {want}, found "
            f"{found['cooperative_matrix']} x{found['cooperative_matrix_muladds']}"
        )
    if facts["subgroup"] is None:
        sys.exit(
            f"{name}: maps work onto subgroups but declares no required subgroup size"
        )

    manifest = {
        "format": 1,
        "name": name,
        "spirv": f"{name}.spv",
        "entry_point": found["entry_point"],
        "spirv_version": found["spirv_version"],
        "memory_model": found["memory_model"],
        "workgroup_size": found["workgroup_size"],
        "dispatch": facts["grid_size"],
        "required_subgroup_size": facts["subgroup"],
        "require_full_subgroups": facts["subgroup"] is not None,
        "push_constants": [],
        "specialization_constants": [],
        "workgroup_variables": found["workgroup_variables"],
        "bindings": found["bindings"],
        "capabilities": found["capabilities"],
        "extensions": found["extensions"],
        "vulkan_features": sorted(
            {FEATURES[c] for c in found["capabilities"] if c in FEATURES}
            | (
                {"subgroupSizeControl", "computeFullSubgroups"}
                if facts["subgroup"]
                else set()
            )
        ),
        "vulkan_extensions": ["VK_KHR_cooperative_matrix"]
        if found["cooperative_matrix"]
        else [],
        "cooperative_matrix": found["cooperative_matrix"],
        "cooperative_matrix_muladds": found["cooperative_matrix_muladds"],
        "fp_rounding_modes": found["fp_rounding_modes"],
        "source": {
            "kernel": f"kernels/gpu/{kernel}.mlir",
            "schedule": f"kernels/gpu/{kernel}.schedule.mlir",
            "target": "kernels/gpu/target.mlir",
            "substitutions": substitutions,
        },
    }
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(
        f"{name}: {os.path.getsize(spv_path)} bytes of SPIR-V, spirv-val clean; "
        f"workgroup {manifest['workgroup_size']}, dispatch {manifest['dispatch']}, "
        f"subgroup {manifest['required_subgroup_size'] or 'any'}; "
        f"{found['cooperative_matrix_muladds']} OpCooperativeMatrixMulAddKHR "
        f"{found['cooperative_matrix']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("kernel", choices=sorted(KERNELS))
    parser.add_argument("--m", type=int, required=True, help="rows of A and C")
    parser.add_argument("--n", type=int, required=True, help="rows of B, columns of C")
    parser.add_argument("--k", type=int, required=True, help="columns of A and B")
    parser.add_argument(
        "--acc", choices=["f16", "f32"], required=True, help="accumulator type"
    )
    args = parser.parse_args()
    for dim, tile in (("m", GEMM_TILE_MN), ("n", GEMM_TILE_MN), ("k", GEMM_TILE_K)):
        if getattr(args, dim) <= 0 or getattr(args, dim) % tile:
            parser.error(
                f"--{dim} must be a positive multiple of the schedule's tile, {tile}"
            )
    build(args)


if __name__ == "__main__":
    main()
