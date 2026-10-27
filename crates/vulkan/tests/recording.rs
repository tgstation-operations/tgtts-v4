//! Recordings on the GPU: a chain of `axpb` dispatches recorded once and submitted many times,
//! against the same dispatches run one at a time and against the CPU. Built with the `device`
//! feature; each test fails on any message of the validation layer.
#![cfg(feature = "device")]

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

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

/// Buffers: 0 x, 1 the two intermediate results, the second at the first offset past the first
/// that the device can bind, 2 the output, 3 the subgroup sizes every link writes. The second
/// link reads and writes ranges of buffer 1.
fn chain(ctx: &Context, kernels: &[Arc<Kernel>], n: usize, timed: bool) -> vulkan::Recording {
    let bytes = 4 * n as u64;
    let after_first = bytes.next_multiple_of(ctx.report().min_storage_buffer_offset_alignment);
    let buffer = |size, residence| {
        ctx.buffer_in(size, residence)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let buffers = vec![
        buffer(bytes, Residence::Upload),
        buffer(after_first + bytes, Residence::Device),
        buffer(bytes, Residence::Readback),
        buffer(bytes, Residence::Readback),
    ];
    let first = Binding::range(1, 0, bytes);
    let second = Binding::range(1, after_first, bytes);
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
    assert_eq!(recording.dispatches(0), 3);
    let submissions = 200;
    let start = Instant::now();
    for step in 0..submissions {
        // A value the host changes between submissions, as a position would be.
        let x = inputs(n, step + 1);
        recording.buffer_mut(0).write(&x).unwrap();
        // Waited for by blocking at once, or by polling first for longer than the chain takes.
        let spin = if step % 2 == 0 { 0 } else { 50 };
        ctx.set_fence_spin(Duration::from_millis(spin));
        // Submitted and waited for at once, started and then waited for, or started and
        // dropped, which waits.
        if step % 3 == 0 {
            recording.submit().unwrap_or_else(|e| panic!("{e}"));
        } else {
            let started = recording.start().unwrap_or_else(|e| panic!("{e}"));
            if step % 3 == 1 {
                started.wait().unwrap_or_else(|e| panic!("{e}"));
            }
        }
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
fn programs_over_shared_buffers_run_alternately_each_as_its_own_dispatches_would() {
    let ctx = context();
    let kernels: Vec<_> = LINKS.iter().map(|&(a, b)| axpb(&ctx, a, b)).collect();
    let n = 10_007;
    let bytes = 4 * n as u64;
    // The first program, the chain, from x in buffer 0 to buffer 2. The second takes the chain's
    // output on through its second and first links, into a buffer added for it, by way of the
    // range of buffer 1 the chain's first link writes.
    let mut recording = chain(&ctx, &kernels, n, false);
    let out = recording
        .add_buffer(
            ctx.buffer_in(bytes, Residence::Readback)
                .unwrap_or_else(|e| panic!("{e}")),
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let scratch = Binding::range(1, 0, bytes);
    let links = [
        (1, Binding::whole(2), scratch),
        (0, scratch, Binding::whole(out)),
    ];
    let dispatches: Vec<_> = links
        .iter()
        .map(|&(link, from, to)| Dispatch {
            kernel: Arc::clone(&kernels[link]),
            bindings: vec![from, to, Binding::whole(3)],
            groups: [groups_for(n as u64, WIDTH) as u32, 1, 1],
            barrier: true,
        })
        .collect();
    let second = recording
        .add_program(&dispatches, true)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((second, recording.programs()), (1, 2));
    assert_eq!(recording.dispatches(second), 2);
    let on = |x: &[f32]| -> Vec<f32> {
        let link = |v: f32, (a, b): (f32, f32)| a * v + b;
        x.iter()
            .map(|&v| link(link(v, LINKS[1]), LINKS[0]))
            .collect()
    };

    for step in 0..6 {
        let x = inputs(n, step + 1);
        recording.buffer_mut(0).write(&x).unwrap();
        recording.submit().unwrap_or_else(|e| panic!("{e}"));
        assert!(
            recording.timings().unwrap().is_none(),
            "the chain is untimed"
        );
        let chained = recording.buffer(2).read::<f32>().unwrap();
        assert_eq!(differences(&chained, &on_the_cpu(&x)), 0, "step {step}");
        // Twice on some steps: a program may follow itself as well as the other.
        for _ in 0..1 + step % 2 {
            recording
                .submit_program(second)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let timings = recording.timings().unwrap().expect("a timed program");
        assert_eq!(timings.len(), 2);
        let got = recording.buffer(out).read::<f32>().unwrap();
        assert_eq!(differences(&got, &on(&chained)), 0, "step {step}");
    }
    assert!(matches!(
        recording.start_program(2),
        Err(Error::Unsupported(_))
    ));

    // The second program recorded again as one link, in place.
    let one = &dispatches[1..];
    recording
        .replace_program(second, one, false)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(recording.timings(), Err(Error::Unsupported(_))));
    recording
        .submit_program(second)
        .unwrap_or_else(|e| panic!("{e}"));
    let scratch: Vec<f32> = recording.buffer(1).read::<f32>().unwrap()[..n].to_vec();
    let link = |&v: &f32| LINKS[0].0 * v + LINKS[0].1;
    let got = recording.buffer(out).read::<f32>().unwrap();
    assert_eq!(
        differences(&got, &scratch.iter().map(link).collect::<Vec<_>>()),
        0
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
        let ticks: u64 = timings.iter().map(|t| t.ticks).sum();
        assert_eq!(recording.elapsed().unwrap().ticks, ticks);
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
    let mut untimed = chain(&ctx, &kernels, n, false);
    assert!(matches!(untimed.elapsed(), Err(Error::Unsupported(_))));
    untimed.buffer_mut(0).write(&x).unwrap();
    let start = Instant::now();
    untimed.submit().unwrap_or_else(|e| panic!("{e}"));
    let wall = start.elapsed().as_nanos() as f64;
    assert!(untimed.timings().unwrap().is_none());
    let gpu = untimed.elapsed().unwrap().nanoseconds;
    assert!(gpu > 0.0 && gpu < wall, "GPU {gpu} ns, wall {wall} ns");
    assert_no_validation_messages(&ctx);
}

/// The host's `CLOCK_MONOTONIC`, the clock a calibration reads beside the GPU's.
fn monotonic_ns() -> f64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a clock every Linux has, and a timespec to write.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as f64 * 1e9 + t.tv_nsec as f64
}

#[test]
fn a_calibration_places_a_submission_between_its_submit_and_its_wait() {
    let ctx = context();
    if ctx.calibrate().unwrap().is_none() {
        eprintln!("the device offers no calibrated timestamps; nothing to check");
        return;
    }
    let kernels: Vec<_> = LINKS.iter().map(|&(a, b)| axpb(&ctx, a, b)).collect();
    let mut recording = chain(&ctx, &kernels, 1 << 22, false);
    for _ in 0..3 {
        let calibration = ctx.calibrate().unwrap().unwrap();
        let before = monotonic_ns();
        recording.submit().unwrap_or_else(|e| panic!("{e}"));
        let after = monotonic_ns();
        let [start, end] = recording
            .ends()
            .unwrap()
            .map(|t| calibration.monotonic_at(t));
        let slack = calibration.deviation_ns as f64;
        assert!(
            before - slack <= start && start < end && end <= after + slack,
            "the GPU from {start} to {end}, the host from {before} to {after}, deviation {slack}"
        );
        eprintln!(
            "{:.1} us from submitting to the GPU's start, {:.1} us on the GPU, {:.1} us from its \
             end to the wait's return; deviation {:.1} us (smoke, not a measurement)",
            (start - before) / 1e3,
            (end - start) / 1e3,
            (after - end) / 1e3,
            slack / 1e3
        );
    }
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
