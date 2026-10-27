//! The cooperative-matrix GEMM from `nix build .#gpu-kernels` on the GPU, against C = A B^T in f64
//! over the same f16 operands. Built with the `device` feature; the kernels are read from the
//! directory `VULKAN_KERNELS` names, which `nix run .#device-tests` sets, and are compiled for
//! RADV on the Radeon 8060S.
//!
//! The kernels index their buffers by the shapes they were compiled for and never check bounds,
//! so each manifest is checked against the buffers and grid before anything reaches the driver: a
//! buffer short of its binding would be a page fault in memory the desktop shares.
#![cfg(feature = "device")]

use std::{ffi::CString, path::PathBuf, thread};

use half::f16;
use serde_json::Value;
use vulkan::{
    Context, DeviceReport, Kernel, KernelDesc,
    executable::count_mnemonics,
    report::{component_type_name, scope_name},
    select::Features,
};

/// The subgroup size the schedule maps the GEMM's work onto.
const SUBGROUP: u32 = 64;
/// The schedule's tile of C per workgroup.
const TILE: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Acc {
    F16,
    F32,
}

impl Acc {
    fn name(self) -> &'static str {
        match self {
            Acc::F16 => "f16",
            Acc::F32 => "f32",
        }
    }

    fn bytes(self) -> usize {
        match self {
            Acc::F16 => 2,
            Acc::F32 => 4,
        }
    }

    /// The unit of rounding, or of truncation, of one operation in the accumulator.
    fn ulp(self) -> f64 {
        match self {
            Acc::F16 => 2f64.powi(-10),
            Acc::F32 => 2f64.powi(-23),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
}

fn context() -> Context {
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    if !ctx.validating() {
        eprintln!("the validation layer is off for this run; `nix run .#device-tests` turns it on");
    }
    ctx
}

fn assert_no_validation_messages(ctx: &Context) {
    let messages = ctx.validation_messages();
    assert!(
        messages.is_empty(),
        "the validation layer reported:\n{}",
        messages.join("\n")
    );
}

/// A kernel's SPIR-V and its manifest.
struct Module {
    name: String,
    spirv: Vec<u8>,
    manifest: Value,
}

fn module(shape: Shape, acc: Acc) -> Module {
    let dir = PathBuf::from(
        std::env::var_os("VULKAN_KERNELS")
            .expect("VULKAN_KERNELS names no directory; `nix run .#device-tests` sets it"),
    );
    let name = format!("gemm_{}acc_{}x{}x{}", acc.name(), shape.m, shape.n, shape.k);
    let read = |file: String| {
        std::fs::read(dir.join(&file))
            .unwrap_or_else(|e| panic!("{}: {e}", dir.join(file).display()))
    };
    let manifest = serde_json::from_slice(&read(format!("{name}.json")))
        .unwrap_or_else(|e| panic!("{name}.json: {e}"));
    Module {
        spirv: read(format!("{name}.spv")),
        name,
        manifest,
    }
}

fn uints(v: &Value) -> Vec<u64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default()
}

fn strings(v: &Value) -> Vec<&str> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

/// Checks the manifest against what the test binds and dispatches, against the features the
/// runtime enables, and against the cooperative matrices the device offers; returns the grid.
fn check_manifest(module: &Module, shape: Shape, acc: Acc, report: &DeviceReport) -> [u32; 3] {
    let Shape { m, n, k } = shape;
    let manifest = &module.manifest;
    let want = [
        ("a", "read", "f16", [m, k], 2),
        ("b", "read", "f16", [n, k], 2),
        ("c", "write", acc.name(), [m, n], acc.bytes()),
    ];
    let bindings = manifest["bindings"].as_array().expect("bindings");
    assert_eq!(bindings.len(), want.len(), "{}: bindings", module.name);
    for (i, (b, (name, access, element, dims, bytes))) in bindings.iter().zip(want).enumerate() {
        assert_eq!(
            b["binding"].as_u64(),
            Some(i as u64),
            "{}: binding {name}",
            module.name
        );
        assert_eq!(
            b["set"].as_u64(),
            Some(0),
            "{}: binding {name}",
            module.name
        );
        assert_eq!(
            b["name"].as_str(),
            Some(name),
            "{}: binding {i}",
            module.name
        );
        assert_eq!(
            b["access"].as_str(),
            Some(access),
            "{}: binding {name}",
            module.name
        );
        assert_eq!(
            b["element"].as_str(),
            Some(element),
            "{}: binding {name}",
            module.name
        );
        assert_eq!(
            uints(&b["shape"]),
            dims.map(|d| d as u64),
            "{}: binding {name}",
            module.name
        );
        assert_eq!(
            b["bytes"].as_u64(),
            Some((dims[0] * dims[1] * bytes) as u64),
            "{}: binding {name}",
            module.name
        );
    }
    let grid = [m / TILE, n / TILE, 1].map(|g| g as u64);
    assert_eq!(
        uints(&manifest["dispatch"]),
        grid,
        "{}: dispatch",
        module.name
    );
    assert_eq!(
        uints(&manifest["workgroup_size"]),
        [256, 1, 1],
        "{}",
        module.name
    );
    assert_eq!(
        manifest["required_subgroup_size"].as_u64(),
        Some(SUBGROUP.into())
    );
    assert_eq!(manifest["require_full_subgroups"].as_bool(), Some(true));
    for key in ["push_constants", "specialization_constants"] {
        assert!(
            manifest[key].as_array().is_some_and(Vec::is_empty),
            "{}: {key}",
            module.name
        );
    }

    let enabled: Vec<&str> = Features::ALL
        .list()
        .into_iter()
        .filter_map(|(name, on)| on.then_some(name))
        .collect();
    for feature in strings(&manifest["vulkan_features"]) {
        assert!(
            enabled.contains(&feature),
            "{}: needs {feature}, not enabled",
            module.name
        );
    }
    for extension in strings(&manifest["vulkan_extensions"]) {
        assert!(
            vulkan::select::REQUIRED_EXTENSIONS
                .iter()
                .any(|e| e.to_bytes() == extension.as_bytes()),
            "{}: needs {extension}, not enabled",
            module.name
        );
    }
    let offered = |coop: &Value| {
        report.cooperative_matrix.iter().any(|c| {
            let sizes = [("m", c.m), ("n", c.n), ("k", c.k)];
            let types = [("a", c.a), ("b", c.b), ("c", c.c), ("result", c.result)];
            sizes
                .iter()
                .all(|&(d, v)| coop[d].as_u64() == Some(v.into()))
                && types
                    .iter()
                    .all(|&(t, v)| coop[t].as_str() == Some(component_type_name(v).as_ref()))
                && coop["scope"].as_str() == Some(scope_name(c.scope).as_ref())
                && !c.saturating
        })
    };
    let coops = manifest["cooperative_matrix"]
        .as_array()
        .expect("cooperative_matrix");
    assert!(!coops.is_empty(), "{}: no cooperative matrix", module.name);
    for coop in coops {
        assert!(
            offered(coop),
            "{}: multiplies at {coop}, which the device does not offer",
            module.name
        );
    }
    grid.map(|g| u32::try_from(g).expect("a grid in 32 bits"))
}

fn kernel(ctx: &Context, module: &Module, subgroup: u32) -> Kernel {
    let entry = CString::new(
        module.manifest["entry_point"]
            .as_str()
            .expect("an entry point"),
    )
    .expect("an entry point without NUL");
    let desc = KernelDesc {
        entry: &entry,
        storage_buffers: 3,
        subgroup_size: Some(subgroup),
        full_subgroups: true,
        constants: &[],
    };
    ctx.kernel(&module.spirv, &desc)
        .unwrap_or_else(|e| panic!("{}: {e}", module.name))
}

/// What disagrees between how the driver compiled `kernel` and how its manifest says it must run:
/// at the subgroup size the schedule divides by, with as many `v_wmma` instructions of the
/// accumulator's type as the module has cooperative-matrix multiply-adds.
fn compiled_mismatch(kernel: &Kernel, module: &Module, acc: Acc) -> Option<String> {
    let executables = kernel.executables().unwrap_or_else(|e| panic!("{e}"));
    let [executable] = executables.as_slice() else {
        return Some(format!(
            "{} executables for one compute shader",
            executables.len()
        ));
    };
    if executable.subgroup_size != SUBGROUP {
        return Some(format!("compiled at wave{}", executable.subgroup_size));
    }
    let Some(assembly) = executable
        .representations
        .iter()
        .find(|r| r.name.contains("Assembly"))
        .and_then(|r| r.text.as_deref())
    else {
        return Some("no assembly among the driver's representations".to_string());
    };
    let counts = count_mnemonics(assembly, "v_wmma_");
    let want = format!("v_wmma_{}_16x16x16_f16", acc.name());
    let total: usize = counts.values().sum();
    let muladds = module.manifest["cooperative_matrix_muladds"].as_u64();
    (counts.keys().any(|m| *m != want) || Some(total as u64) != muladds)
        .then(|| format!("{counts:?} in the assembly, for {muladds:?} multiply-adds of {want}"))
}

/// `len` f16 values of standard deviation about `scale`, from a fixed xorshift sequence.
fn values(len: usize, seed: u32, scale: f32) -> Vec<f16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let uniform = state as f32 / u32::MAX as f32 * 2.0 - 1.0;
            f16::from_f32(uniform * scale * 3f32.sqrt())
        })
        .collect()
}

/// C = A B^T in f64, each product of two f16 values being exact in it, with each output's sum of
/// the magnitudes of its products, the scale its rounding error is bounded by.
fn reference(a: &[f16], b: &[f16], shape: Shape) -> (Vec<f64>, Vec<f64>) {
    let Shape { m, n, k } = shape;
    let a: Vec<f64> = a.iter().map(|v| v.to_f64()).collect();
    let b: Vec<f64> = b.iter().map(|v| v.to_f64()).collect();
    let mut value = vec![0f64; m * n];
    let mut magnitude = vec![0f64; m * n];
    let threads = thread::available_parallelism()
        .map_or(1, |t| t.get())
        .min(m);
    let rows = m.div_ceil(threads);
    thread::scope(|s| {
        for (chunk, (values, magnitudes)) in value
            .chunks_mut(rows * n)
            .zip(magnitude.chunks_mut(rows * n))
            .enumerate()
        {
            let (a, b) = (&a, &b);
            s.spawn(move || {
                for (r, (vs, ms)) in values
                    .chunks_mut(n)
                    .zip(magnitudes.chunks_mut(n))
                    .enumerate()
                {
                    let row = &a[(chunk * rows + r) * k..][..k];
                    for (j, (v, mag)) in vs.iter_mut().zip(ms).enumerate() {
                        let products = row.iter().zip(&b[j * k..][..k]).map(|(x, y)| x * y);
                        (*v, *mag) = products.fold((0.0, 0.0), |(s, m), p| (s + p, m + p.abs()));
                    }
                }
            });
        }
    });
    (value, magnitude)
}

/// How far the device's C is from the reference.
#[derive(Debug)]
struct Errors {
    /// Outputs past the bound of K sequential operations each off by up to one unit of the
    /// accumulator, and a last rounding of the output.
    past_bound: usize,
    /// The output with the largest error relative to its bound, and that ratio.
    worst: (usize, f64),
    /// RMS of the errors over RMS of the reference outputs.
    relative_rms: f64,
    /// The column of C whose relative RMS error is the largest, and that error: one wrong column
    /// is lost in the RMS of a wide matrix, and an f16 accumulator's bound on any one output is
    /// too loose to see it.
    worst_column: (usize, f64),
}

fn compare(device: &[f64], value: &[f64], magnitude: &[f64], shape: Shape, acc: Acc) -> Errors {
    let k = shape.k;
    let mut errors = Errors {
        past_bound: 0,
        worst: (0, 0.0),
        relative_rms: 0.0,
        worst_column: (0, 0.0),
    };
    let (mut err2, mut ref2) = (0.0, 0.0);
    let (mut column_err2, mut column_ref2) = (vec![0f64; shape.n], vec![0f64; shape.n]);
    for (i, ((&d, &v), &mag)) in device.iter().zip(value).zip(magnitude).enumerate() {
        let error = (d - v).abs();
        let ratio = error / (k as f64 * acc.ulp() * mag + acc.ulp() * v.abs() + f64::MIN_POSITIVE);
        if ratio > 1.0 || !d.is_finite() {
            errors.past_bound += 1;
        }
        if ratio > errors.worst.1 || !d.is_finite() {
            errors.worst = (i, ratio);
        }
        err2 += error * error;
        ref2 += v * v;
        column_err2[i % shape.n] += error * error;
        column_ref2[i % shape.n] += v * v;
    }
    errors.relative_rms = (err2 / ref2).sqrt();
    for (j, (e, r)) in column_err2.iter().zip(&column_ref2).enumerate() {
        let rms = (e / r).sqrt();
        if rms.is_nan() || rms > errors.worst_column.1 {
            errors.worst_column = (j, rms);
        }
    }
    errors
}

/// The relative RMS error an f16 accumulator may reach at `k`: four times 1.05e-4 sqrt(K), the
/// growth measured of f16-accumulating cooperative-matrix GEMMs on this GPU.
fn f16_rms_bound(k: usize) -> f64 {
    4.0 * 1.05e-4 * (k as f64).sqrt()
}

/// One GEMM pipeline, ready to dispatch.
struct Gemm {
    kernel: Kernel,
    grid: [u32; 3],
    shape: Shape,
    acc: Acc,
}

struct Run {
    c: Vec<f64>,
    nanoseconds: Vec<f64>,
}

/// Runs the GEMM `reps` times over `a` and `b`, and returns C and each dispatch's GPU time.
fn run(ctx: &Context, gemm: &Gemm, a: &[f16], b: &[f16], reps: usize) -> Run {
    let buffer = |len: usize| ctx.buffer(len as u64).unwrap_or_else(|e| panic!("{e}"));
    let (mut ab, mut bb) = (buffer(2 * a.len()), buffer(2 * b.len()));
    let mut cb = buffer(gemm.acc.bytes() * gemm.shape.m * gemm.shape.n);
    ab.write(&a.iter().map(|v| v.to_bits()).collect::<Vec<u16>>())
        .unwrap();
    bb.write(&b.iter().map(|v| v.to_bits()).collect::<Vec<u16>>())
        .unwrap();
    let nanoseconds = (0..reps)
        .map(|_| {
            ctx.dispatch(&gemm.kernel, &mut [&mut ab, &mut bb, &mut cb], gemm.grid)
                .unwrap_or_else(|e| panic!("{e}"))
                .nanoseconds
        })
        .collect();
    let c = match gemm.acc {
        Acc::F16 => cb
            .read::<u16>()
            .unwrap()
            .into_iter()
            .map(|h| f16::from_bits(h).to_f64())
            .collect(),
        Acc::F32 => cb
            .read::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect(),
    };
    Run { c, nanoseconds }
}

/// The GEMM at `shape` on the device against the reference, the check run against a planted
/// defect, and the compiled pipeline against its manifest. Returns each dispatch's GPU time.
fn gemm(ctx: &Context, shape: Shape, acc: Acc, reps: usize) -> Vec<f64> {
    let module = module(shape, acc);
    let grid = check_manifest(&module, shape, acc, ctx.report());
    let kernel = kernel(ctx, &module, SUBGROUP);
    if let Some(why) = compiled_mismatch(&kernel, &module, acc) {
        panic!("{}: {why}", module.name);
    }
    let gemm = Gemm {
        kernel,
        grid,
        shape,
        acc,
    };

    let a = values(shape.m * shape.k, 0x9e37_79b9, 0.5);
    let b = values(shape.n * shape.k, 0x85eb_ca6b, 0.05);
    let (value, magnitude) = reference(&a, &b, shape);
    let passes = |errors: &Errors| {
        errors.past_bound == 0
            && (acc == Acc::F32 || errors.worst_column.1 <= f16_rms_bound(shape.k))
    };

    let good = run(ctx, &gemm, &a, &b, reps);
    let errors = compare(&good.c, &value, &magnitude, shape, acc);
    assert!(passes(&errors), "{}: {errors:?}", module.name);

    // The defect the check is for: one row of B negated on the device, so one column of C is.
    let column = shape.n / 3;
    let mut flipped = b.clone();
    for v in &mut flipped[column * shape.k..][..shape.k] {
        *v = -*v;
    }
    let bad = run(ctx, &gemm, &a, &flipped, 1);
    let errors_bad = compare(&bad.c, &value, &magnitude, shape, acc);
    assert!(
        !passes(&errors_bad),
        "{}: a negated row of B passed: {errors_bad:?}",
        module.name
    );
    assert_eq!(
        errors_bad.worst_column.0, column,
        "{}: {errors_bad:?}",
        module.name
    );

    eprintln!(
        "{}: relative RMS error {:.2e}, {:.2e} in the worst column, worst output at {:.2} of its \
         bound; with a row of B negated its column's is {:.2e}",
        module.name,
        errors.relative_rms,
        errors.worst_column.1,
        errors.worst.1,
        errors_bad.worst_column.1
    );
    good.nanoseconds
}

#[test]
fn small_gemms_agree_with_the_reference_at_either_accumulator() {
    let ctx = context();
    assert!(
        ctx.captures_executables(),
        "the device does not report what it compiled"
    );
    for shape in [
        Shape {
            m: 128,
            n: 128,
            k: 64,
        },
        Shape {
            m: 256,
            n: 384,
            k: 512,
        },
    ] {
        for acc in [Acc::F16, Acc::F32] {
            gemm(&ctx, shape, acc, 1);
        }
    }
    assert_no_validation_messages(&ctx);
}

#[test]
fn a_gemm_compiled_at_another_wave_size_is_caught() {
    // Pairs of 32-wide subgroups would compute the same tiles and write the right numbers, so
    // only the driver's account of the pipeline shows it; it is never dispatched.
    let ctx = context();
    let shape = Shape {
        m: 128,
        n: 128,
        k: 64,
    };
    for acc in [Acc::F16, Acc::F32] {
        let module = module(shape, acc);
        let why = compiled_mismatch(&kernel(&ctx, &module, 32), &module, acc);
        assert_eq!(
            why.as_deref(),
            Some("compiled at wave32"),
            "{}",
            module.name
        );
    }
    assert_no_validation_messages(&ctx);
}

#[test]
fn a_prefill_sized_projection_agrees_with_the_reference_and_is_timed() {
    // The base LM's up projection over a 446-position prompt, padded to the schedule's 128.
    let ctx = context();
    let shape = Shape {
        m: 512,
        n: 6144,
        k: 2048,
    };
    let flops = 2.0 * (shape.m * shape.n * shape.k) as f64;
    for acc in [Acc::F16, Acc::F32] {
        let mut ns = gemm(&ctx, shape, acc, 11);
        // The first dispatch warms caches and clocks.
        ns.remove(0);
        ns.sort_by(f64::total_cmp);
        eprintln!(
            "{}acc {}x{}x{}: GPU time min {:.3} ms, median {:.3} ms, max {:.3} ms over {} \
             dispatches; {:.2} TFLOP/s at the median",
            acc.name(),
            shape.m,
            shape.n,
            shape.k,
            ns[0] / 1e6,
            ns[ns.len() / 2] / 1e6,
            ns[ns.len() - 1] / 1e6,
            ns.len(),
            flops / ns[ns.len() / 2] / 1e3
        );
    }
    assert_no_validation_messages(&ctx);
}
