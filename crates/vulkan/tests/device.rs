//! The runtime on the GPU, with a hand-written kernel. Built with the `device` feature; each test
//! fails on any message the validation layer reports, which `nix run .#device-tests` attaches.
//! The timings printed are one dispatch each, never a measurement.
#![cfg(feature = "device")]

use std::time::Instant;

use ash::vk;
use vulkan::{
    Context, Error, KernelDesc, SpecValue, groups_for,
    kernel::required_subgroup_size_allowed,
    memory::{STORAGE_PREFERENCE, choose_memory_type},
};

/// `y[i] = a * x[i] + b; s[i] = SubgroupSize` (source and build command in
/// `kernels/axpb.spvasm`).
const AXPB: &[u8] = include_bytes!("kernels/axpb.spv");

/// The specialization constants' SpecIds in `axpb`.
const WIDTH: u32 = 0;
const A: u32 = 1;
const B: u32 = 2;

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

/// `n` floats with full mantissas, both signs, over a few binades, from a fixed xorshift
/// sequence: products with `a` and sums with `b` then round in most elements, so a fused
/// multiply-add would show as a last-bit difference.
fn inputs(n: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Sign, an exponent in [-4, 3], and 23 random mantissa bits.
            let sign = state & 0x8000_0000;
            let exponent = 123 + (state >> 23) % 8;
            f32::from_bits(sign | exponent << 23 | (state & 0x7f_ffff))
        })
        .collect()
}

struct Run {
    y: Vec<f32>,
    s: Vec<u32>,
    nanoseconds: f64,
}

/// Runs `axpb` over `x` with workgroups of `width`, the given required subgroup size
/// (None: the driver's choice), and the spare element past the end of y and s seeded with
/// a sentinel to show the kernel stops at the length of x.
fn run_axpb(
    ctx: &Context,
    x: &[f32],
    width: u32,
    subgroup_size: Option<u32>,
    a: f32,
    b: f32,
) -> Run {
    let constants = [
        (WIDTH, SpecValue::U32(width)),
        (A, SpecValue::F32(a)),
        (B, SpecValue::F32(b)),
    ];
    let desc = KernelDesc {
        storage_buffers: 3,
        subgroup_size,
        full_subgroups: subgroup_size.is_some(),
        constants: &constants,
        ..KernelDesc::default()
    };
    let kernel = ctx.kernel(AXPB, &desc).unwrap_or_else(|e| panic!("{e}"));
    let n = x.len() as u64;
    let buffer = |bytes| ctx.buffer(bytes).unwrap_or_else(|e| panic!("{e}"));
    let (mut xs, mut ys, mut ss) = (buffer(4 * n), buffer(4 * (n + 1)), buffer(4 * (n + 1)));
    xs.write(x);
    let mut sentinel = vec![0f32; x.len()];
    sentinel.push(-12_345.0);
    ys.write(&sentinel);
    let groups = u32::try_from(groups_for(n, width))
        .unwrap_or_else(|_| panic!("{n} elements is too many workgroups"));
    let timing = ctx
        .dispatch(&kernel, &mut [&mut xs, &mut ys, &mut ss], [groups, 1, 1])
        .unwrap_or_else(|e| panic!("{e}"));
    let mut y = ys.read::<f32>();
    let mut s = ss.read::<u32>();
    assert_eq!(
        y.pop(),
        Some(-12_345.0),
        "the kernel wrote past the length of x"
    );
    assert_eq!(s.pop(), Some(0), "the kernel wrote past the length of x");
    Run {
        y,
        s,
        nanoseconds: timing.nanoseconds,
    }
}

#[test]
fn the_device_offers_what_the_runtime_relies_on() {
    let ctx = context();
    let r = ctx.report();
    eprintln!(
        "device: {} ({}), queue family {}",
        r.name,
        r.driver_info,
        ctx.queue_family()
    );

    let has = |c: i32, result: i32| {
        r.cooperative_matrix.iter().any(|m| {
            (m.m, m.n, m.k, m.a, m.b, m.c, m.result, m.scope) == (16, 16, 16, 0, 0, c, result, 3)
        })
    };
    assert!(
        has(1, 1),
        "no 16x16x16 f16 x f16 + f32 cooperative matrix at subgroup scope"
    );
    assert!(
        has(0, 0),
        "no 16x16x16 f16 x f16 + f16 cooperative matrix at subgroup scope"
    );

    for size in [32, 64] {
        assert!(
            required_subgroup_size_allowed(
                size,
                r.features.subgroup_size_control,
                (r.min_subgroup_size, r.max_subgroup_size),
                r.required_subgroup_size_stages,
            ),
            "a compute pipeline cannot require a subgroup size of {size}"
        );
    }
    assert!(r.timestamp_period > 0.0);

    let chosen =
        choose_memory_type(&r.types, &r.heaps, u32::MAX, 1 << 20).expect("a storage memory type");
    assert!(
        r.types[chosen as usize]
            .flags
            .contains(STORAGE_PREFERENCE[0]),
        "no device-local, host-visible memory type"
    );
    let buffer = ctx.buffer(1 << 20).unwrap();
    assert_eq!(buffer.memory_type(), chosen);
    assert!(
        r.types[buffer.memory_type() as usize]
            .flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
        "buffers are not in device-local memory"
    );
    assert_no_validation_messages(&ctx);
}

#[test]
fn axpb_matches_the_cpu_bit_for_bit_at_each_required_subgroup_size() {
    let ctx = context();
    let (a, b) = (1.7f32, -0.3f32);
    // Not a multiple of the workgroup, so the last workgroup has spare invocations.
    let x = inputs(100_003);
    let want: Vec<f32> = x.iter().map(|&v| a * v + b).collect();
    for size in [32, 64] {
        let run = run_axpb(&ctx, &x, 64, Some(size), a, b);
        let wrong: Vec<usize> = (0..x.len())
            .filter(|&i| run.y[i].to_bits() != want[i].to_bits())
            .collect();
        assert!(
            wrong.is_empty(),
            "wave{size}: {} of {} elements differ from the CPU; first at {}: x {:e}, GPU {:e}, CPU {:e}",
            wrong.len(),
            x.len(),
            wrong[0],
            x[wrong[0]],
            run.y[wrong[0]],
            want[wrong[0]]
        );
        assert!(
            run.s.iter().all(|&s| s == size),
            "wave{size} required, but the kernel saw subgroup sizes {:?}",
            run.s.iter().collect::<std::collections::BTreeSet<_>>()
        );
        eprintln!(
            "wave{size}: {} elements bit-exact; {:.1} us of GPU time (smoke, not a measurement)",
            x.len(),
            run.nanoseconds / 1e3
        );
    }
    assert_no_validation_messages(&ctx);
}

#[test]
fn left_to_the_driver_the_wave_is_64_except_for_workgroups_of_32_or_fewer() {
    // RADV's rule for a SPIR-V 1.6 module with no required size: wave32 for a workgroup
    // of at most 32 invocations, its default (wave64) otherwise. If this fails, the
    // driver's rule changed.
    let ctx = context();
    let x = inputs(4_096);
    for (width, expected) in [(16, 32), (32, 32), (64, 64), (128, 64), (256, 64)] {
        let run = run_axpb(&ctx, &x, width, None, 1.0, 0.0);
        let seen: std::collections::BTreeSet<_> = run.s.iter().copied().collect();
        eprintln!("workgroup {width}: subgroup sizes {seen:?}");
        assert_eq!(seen, [expected].into(), "workgroup {width}");
    }
    assert_no_validation_messages(&ctx);
}

#[test]
fn a_subgroup_size_the_device_cannot_require_is_refused() {
    let ctx = context();
    for size in [16, 48, 128] {
        let desc = KernelDesc {
            storage_buffers: 3,
            subgroup_size: Some(size),
            ..KernelDesc::default()
        };
        assert!(
            matches!(ctx.kernel(AXPB, &desc), Err(Error::Unsupported(_))),
            "size {size}"
        );
    }
    assert_no_validation_messages(&ctx);
}

#[test]
fn timestamps_grow_with_the_work_and_never_exceed_the_wall_clock() {
    // A check of the instrument by a differently shaped stimulus: 16x the elements must
    // take longer on the GPU's clock, and the GPU's interval sits inside the CPU's.
    let ctx = context();
    let mut gpu = Vec::new();
    for n in [1 << 20, 1 << 24] {
        let x = inputs(n);
        let start = Instant::now();
        let run = run_axpb(&ctx, &x, 64, Some(64), 1.0, 0.0);
        let wall = start.elapsed().as_nanos() as f64;
        assert!(run.nanoseconds > 0.0, "a zero interval for {n} elements");
        assert!(
            run.nanoseconds < wall,
            "GPU {} ns > wall {wall} ns for {n} elements",
            run.nanoseconds
        );
        // 4 bytes read and 8 written per element.
        let bytes = 12.0 * n as f64;
        eprintln!(
            "{n} elements: {:.1} us GPU, {:.1} GB/s moved (smoke, not a measurement)",
            run.nanoseconds / 1e3,
            bytes / run.nanoseconds
        );
        gpu.push(run.nanoseconds);
    }
    assert!(gpu[1] > gpu[0], "16x the work took no longer: {gpu:?}");
    assert_no_validation_messages(&ctx);
}

#[test]
fn the_driver_reports_what_it_compiled_each_pipeline_into() {
    // A probe of a different kind from the kernel's own report of its subgroup size: the
    // driver's account of the compiled executable must agree with it at each width.
    let ctx = context();
    assert!(
        ctx.captures_executables(),
        "the device does not offer VK_KHR_pipeline_executable_properties"
    );
    let x = inputs(4_096);
    for size in [32, 64] {
        let constants = [
            (WIDTH, SpecValue::U32(64)),
            (A, SpecValue::F32(1.0)),
            (B, SpecValue::F32(0.0)),
        ];
        let desc = KernelDesc {
            storage_buffers: 3,
            subgroup_size: Some(size),
            full_subgroups: true,
            constants: &constants,
            ..KernelDesc::default()
        };
        let kernel = ctx.kernel(AXPB, &desc).unwrap_or_else(|e| panic!("{e}"));
        let executables = kernel.executables().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            executables.len(),
            1,
            "one executable for one compute shader"
        );
        let e = &executables[0];
        assert_eq!(
            e.subgroup_size, size,
            "wave{size}: the driver reports {}",
            e.subgroup_size
        );
        let vgprs = e.statistic("VGPRs").map(|v| v.as_f64());
        assert!(
            vgprs.is_some_and(|v| v > 0.0),
            "no VGPR count: {:?}",
            e.statistics
        );
        assert!(
            e.text().contains("s_endpgm"),
            "no final assembly among {:?}",
            e.representations
                .iter()
                .map(|r| &r.name)
                .collect::<Vec<_>>()
        );
        let run = run_axpb(&ctx, &x, 64, Some(size), 1.0, 0.0);
        assert!(run.s.iter().all(|&s| s == size), "wave{size}");
        eprintln!("wave{size}: the driver and the kernel agree; {vgprs:?} VGPRs");
    }
    assert_no_validation_messages(&ctx);
}
