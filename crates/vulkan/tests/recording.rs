//! Recordings on the GPU: a chain of `axpb` dispatches recorded once and submitted many times,
//! against the same dispatches run one at a time and against the CPU. Built with the `device`
//! feature; each test fails on any message of the validation layer.
#![cfg(feature = "device")]

use std::{sync::Arc, time::Instant};

use vulkan::{
    Binding, Context, Dispatch, Error, Kernel, KernelDesc, Residence, SpecValue, groups_for,
};

const AXPB: &[u8] = include_bytes!("kernels/axpb.spv");
const WIDTH: u32 = 64;
/// The size every device the kernels run on can require.
const SUBGROUP: u32 = 32;

/// The chain's (a, b): each link computes `a * x + b` of the one before.
const LINKS: [(f32, f32); 3] = [(1.7, -0.3), (-0.6, 0.25), (1.1, 2.0)];

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

fn axpb(ctx: &Context, a: f32, b: f32) -> Arc<Kernel> {
    let constants = [
        (0, SpecValue::U32(WIDTH)),
        (1, SpecValue::F32(a)),
        (2, SpecValue::F32(b)),
    ];
    let desc = KernelDesc {
        storage_buffers: 3,
        subgroup_size: Some(SUBGROUP),
        full_subgroups: true,
        constants: &constants,
        ..KernelDesc::default()
    };
    Arc::new(ctx.kernel(AXPB, &desc).unwrap_or_else(|e| panic!("{e}")))
}

/// `n` floats from a fixed xorshift sequence seeded by `seed`, over a few binades.
fn inputs(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let sign = state & 0x8000_0000;
            let exponent = 123 + (state >> 23) % 8;
            f32::from_bits(sign | exponent << 23 | (state & 0x7f_ffff))
        })
        .collect()
}

fn on_the_cpu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| LINKS.iter().fold(v, |v, &(a, b)| a * v + b))
        .collect()
}

/// Buffers: 0 x, 1 the two intermediate results back to back, 2 the output, 3 the subgroup
/// sizes every link writes. The second link reads and writes ranges of buffer 1.
fn chain(ctx: &Context, kernels: &[Arc<Kernel>], n: usize, timed: bool) -> vulkan::Recording {
    let bytes = 4 * n as u64;
    let buffer = |size, residence| {
        ctx.buffer_in(size, residence)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let buffers = vec![
        buffer(bytes, Residence::Upload),
        buffer(2 * bytes, Residence::Device),
        buffer(bytes, Residence::Readback),
        buffer(bytes, Residence::Readback),
    ];
    let first = Binding::range(1, 0, bytes);
    let second = Binding::range(1, bytes, bytes);
    let groups = [groups_for(n as u64, WIDTH) as u32, 1, 1];
    let links = [
        (Binding::whole(0), first),
        (first, second),
        (second, Binding::whole(2)),
    ];
    let dispatches: Vec<_> = kernels
        .iter()
        .zip(links)
        .map(|(kernel, (from, to))| Dispatch {
            kernel: Arc::clone(kernel),
            bindings: vec![from, to, Binding::whole(3)],
            groups,
            barrier: true,
        })
        .collect();
    ctx.record(buffers, &dispatches, timed)
        .unwrap_or_else(|e| panic!("{e}"))
}

fn differences(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count()
}

#[test]
fn a_recorded_chain_resubmitted_matches_its_dispatches_run_one_at_a_time() {
    let ctx = context();
    assert!(
        ctx.robust_buffer_access2(),
        "the device does not offer robustBufferAccess2"
    );
    let kernels: Vec<_> = LINKS.iter().map(|&(a, b)| axpb(&ctx, a, b)).collect();
    // Not a multiple of the workgroup.
    let n = 10_007;

    // The links one at a time, each a submission of its own.
    let x = inputs(n, 1);
    let buffer = |residence| {
        ctx.buffer_in(4 * n as u64, residence)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let mut values = buffer(Residence::Upload);
    values.write(&x).unwrap();
    let mut sizes = buffer(Residence::Readback);
    let groups = [groups_for(n as u64, WIDTH) as u32, 1, 1];
    for kernel in &kernels {
        let mut next = buffer(Residence::Readback);
        ctx.dispatch(kernel, &mut [&mut values, &mut next, &mut sizes], groups)
            .unwrap_or_else(|e| panic!("{e}"));
        values = next;
    }
    let one_at_a_time = values.read::<f32>().unwrap();
    assert_eq!(differences(&one_at_a_time, &on_the_cpu(&x)), 0);

    let mut recording = chain(&ctx, &kernels, n, false);
    assert_eq!(recording.dispatches(), 3);
    let submissions = 200;
    let start = Instant::now();
    for step in 0..submissions {
        // A value the host changes between submissions, as a position would be.
        let x = inputs(n, step + 1);
        recording.buffer_mut(0).write(&x).unwrap();
        recording.submit().unwrap_or_else(|e| panic!("{e}"));
        let got = recording.buffer(2).read::<f32>().unwrap();
        if step == 0 {
            assert_eq!(differences(&got, &one_at_a_time), 0, "the first submission");
        }
        let wrong = differences(&got, &on_the_cpu(&x));
        assert_eq!(wrong, 0, "submission {step}: {wrong} of {n} differ");
        assert!(
            recording
                .buffer(3)
                .read::<u32>()
                .unwrap()
                .iter()
                .all(|&s| s == SUBGROUP)
        );
    }
    eprintln!(
        "{submissions} submissions of 3 dispatches over {n} elements, bit-exact each time; \
         {:.1} us each with the host's writes and reads (smoke, not a measurement)",
        start.elapsed().as_secs_f64() * 1e6 / f64::from(submissions)
    );
    assert_no_validation_messages(&ctx);
}

#[test]
fn a_timed_recording_times_each_dispatch_inside_the_wall_clock() {
    let ctx = context();
    let kernels: Vec<_> = LINKS.iter().map(|&(a, b)| axpb(&ctx, a, b)).collect();
    let n = 1 << 22;
    let mut recording = chain(&ctx, &kernels, n, true);
    let x = inputs(n, 7);
    recording.buffer_mut(0).write(&x).unwrap();
    for _ in 0..3 {
        let start = Instant::now();
        recording.submit().unwrap_or_else(|e| panic!("{e}"));
        let wall = start.elapsed().as_nanos() as f64;
        let timings = recording.timings().unwrap().expect("a timed recording");
        assert_eq!(timings.len(), 3);
        assert!(timings.iter().all(|t| t.nanoseconds > 0.0), "{timings:?}");
        let gpu: f64 = timings.iter().map(|t| t.nanoseconds).sum();
        assert!(gpu < wall, "GPU {gpu} ns > wall {wall} ns");
        eprintln!(
            "{} us per dispatch, {:.1} us on the GPU in all, {:.1} us submit to fence \
             (smoke, not a measurement)",
            timings
                .iter()
                .map(|t| format!("{:.1}", t.nanoseconds / 1e3))
                .collect::<Vec<_>>()
                .join(" + "),
            gpu / 1e3,
            wall / 1e3
        );
    }
    assert_eq!(
        differences(&recording.buffer(2).read::<f32>().unwrap(), &on_the_cpu(&x)),
        0
    );
    let untimed = chain(&ctx, &kernels, 64, false);
    assert!(untimed.timings().unwrap().is_none());
    assert_no_validation_messages(&ctx);
}

#[test]
fn a_binding_outside_its_buffer_is_refused_before_the_driver_sees_it() {
    let ctx = context();
    let kernel = axpb(&ctx, 1.0, 0.0);
    let buffers = || {
        (0..3)
            .map(|_| ctx.buffer(256).unwrap_or_else(|e| panic!("{e}")))
            .collect::<Vec<_>>()
    };
    let dispatch = |bindings| Dispatch {
        kernel: Arc::clone(&kernel),
        bindings,
        groups: [1, 1, 1],
        barrier: false,
    };
    for bindings in [
        vec![
            Binding::whole(0),
            Binding::range(1, 128, 256),
            Binding::whole(2),
        ],
        vec![Binding::whole(0), Binding::whole(1), Binding::whole(3)],
        vec![Binding::whole(0), Binding::whole(1)],
        vec![
            Binding::whole(0),
            Binding::range(1, 2, 64),
            Binding::whole(2),
        ],
    ] {
        let refused = ctx.record(buffers(), &[dispatch(bindings.clone())], false);
        assert!(
            matches!(refused, Err(Error::Unsupported(_))),
            "{bindings:?}: {refused:?}"
        );
    }
    assert_no_validation_messages(&ctx);
}
