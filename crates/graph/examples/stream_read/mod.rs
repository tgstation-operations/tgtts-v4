//! The floor the timing examples measure against: the rate of a streaming read.

use graph::{
    graph::Pipeline,
    kernels::{Kernels, STREAM_READ},
};
use vulkan::{Context, Residence, SpecValue, memory::STAGING_BYTES};

/// The best rate, in GB/s, at which `stream_read` reads `bytes` of device memory, over 1, 4 and
/// 16 loads per invocation in workgroups of 256, each the best of nine dispatches after one.
pub fn stream_read(ctx: &Context, kernels: &mut Kernels, subgroup: u32, bytes: u64) -> f64 {
    let mut data = ctx
        .buffer_in::<u32>((bytes / 4) as usize, Residence::Device)
        .unwrap_or_else(|e| panic!("{e}"));
    // Words that do not repeat, filled a staging buffer at a time.
    let mut state = 0x2545_f491_u32;
    let chunk = (STAGING_BYTES / 4) as usize;
    for at in (0..bytes / 4).step_by(chunk) {
        let words: Vec<u32> = (0..chunk.min((bytes / 4 - at) as usize))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            })
            .collect();
        data.write_at(at as usize, &words)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    let workgroup = 256;
    let mut best = 0f64;
    for loads in [1, 4, 16] {
        let pipeline = Pipeline {
            kernel: &STREAM_READ,
            constants: vec![
                (0, SpecValue::U32(subgroup)),
                (1, SpecValue::U32(workgroup)),
                (2, SpecValue::U32(loads)),
            ],
            subgroup_size: Some(subgroup),
        };
        let kernel = kernels.get(&pipeline).unwrap_or_else(|e| panic!("{e}"));
        let groups = bytes / 16 / u64::from(workgroup * loads);
        let mut out = ctx
            .buffer_in::<u32>(
                (groups * u64::from(workgroup / subgroup)) as usize,
                Residence::Readback,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        let mut rates = Vec::new();
        for _ in 0..10 {
            let t = ctx
                .dispatch(&kernel, &mut [&mut data, &mut out], [groups as u32, 1, 1])
                .unwrap_or_else(|e| panic!("{e}"));
            rates.push(bytes as f64 / t.nanoseconds);
        }
        let rate = rates[1..].iter().copied().fold(0.0, f64::max);
        println!(
            "stream_read of {} MiB, {loads} loads per invocation: best {rate:.0} GB/s",
            bytes >> 20
        );
        best = best.max(rate);
    }
    best
}
