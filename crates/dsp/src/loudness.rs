//! Loudness as ITU-R BS.1770-4 measures it, the meter EBU R128 normalizes to: the integrated
//! loudness of a single channel in LUFS, and its true peak in dBTP, full scale at 1; and speech
//! brought to a loudness by it.

use std::f64::consts::PI;

use crate::{limiter::limit, segment::Segment};

/// EBU R128's highest true peak, in dBTP.
pub const CEILING: f64 = -1.0;

/// The integrated loudness under which [`normalize`] takes audio for the silence it nearly is,
/// in LUFS: brought to the target, a model's near-silent output is its noise and clicks raised
/// by tens of decibels, or by hundreds where the limiter holds the clicks down.
pub const FLOOR: f64 = -60.0;

/// How near the target [`normalize`] brings the loudness, in LU, where the limiter has lowered it.
const NEAR: f64 = 0.05;
/// The most gains [`normalize`] tries.
const TRIES: usize = 8;

/// What [`normalize`] made of speech.
#[derive(Clone, Debug, PartialEq)]
pub struct Normalized {
    /// The speech.
    pub segment: Segment,
    /// The gain it took before the limiter, in dB.
    pub gain: f64,
    /// The most the limiter took off, in dB: 0 where it did nothing.
    pub reduction: f64,
    /// The fraction of the samples whose gain the limiter lowered.
    pub limited: f64,
}

/// `segment` brought to an integrated loudness of `target` LUFS by a gain of at most `most` dB,
/// and its true peak held at or under `ceiling` dBTP by the [limiter](crate::limiter). Where the
/// limiter takes enough off to lower the loudness, the gain is raised and the limiter run again
/// until the loudness is within 0.05 LU of the target, or the gain is `most`. Anything quieter
/// than [`FLOOR`], silence among it, is left as it is.
pub fn normalize(segment: &Segment, target: f64, ceiling: f64, most: f64) -> Normalized {
    let samples: Vec<f64> = (segment.samples.iter())
        .map(|&s| f64::from(s) / 32768.0)
        .collect();
    let loudness = integrated(&samples, segment.rate);
    if loudness < FLOOR {
        return Normalized {
            segment: segment.clone(),
            gain: 0.0,
            reduction: 0.0,
            limited: 0.0,
        };
    }
    let mut gain = (target - loudness).min(most);
    let mut last: Option<(f64, f64)> = None;
    let mut tries = 0;
    let limited = loop {
        let factor = 10f64.powf(gain / 20.0);
        let scaled: Vec<f64> = samples.iter().map(|s| s * factor).collect();
        let limited = limit(&scaled, segment.rate, ceiling);
        let reached = integrated(&limited.samples, segment.rate);
        tries += 1;
        let capped = gain == most && reached < target;
        if (reached - target).abs() <= NEAR || tries == TRIES || capped {
            break limited;
        }
        // Where the limiter holds peaks down, the loudness rises by less than the gain: the next
        // gain follows its rise between the last two tries.
        let rise = last.map_or(1.0, |(g, r)| (reached - r) / (gain - g));
        last = Some((gain, reached));
        gain = (gain + (target - reached) / rise.clamp(0.1, 1.0)).min(most);
    };
    let pcm = limited.samples.iter().map(|s| (s * 32768.0).round() as i16);
    Normalized {
        segment: Segment::new(segment.rate, pcm.collect()),
        gain,
        reduction: limited.reduction,
        limited: limited.fraction,
    }
}

/// The integrated loudness of `samples` at `rate` Hz, in LUFS: the mean square of their
/// K-weighted values over 400 ms blocks overlapping by 75%, leaving out the blocks under
/// -70 LUFS and then those more than 10 LU under the loudness of the rest, or `-inf` where no
/// block is left. Samples shorter than a block are measured as one block of their own length,
/// where the standard has no measure for them.
pub fn integrated(samples: &[f64], rate: u32) -> f64 {
    let weighted = k_weighted(samples, rate);
    // Rounded as libebur128 rounds them, the standard asking only for the nearest sample.
    let step = (rate as usize + 5) / 10;
    let block = 4 * step;
    let blocks: Vec<f64> = if weighted.is_empty() {
        Vec::new()
    } else if weighted.len() < block {
        vec![mean_square(&weighted)]
    } else {
        (0..=(weighted.len() - block) / step)
            .map(|j| mean_square(&weighted[j * step..j * step + block]))
            .collect()
    };
    let gated = |threshold: f64| -> Vec<f64> {
        let kept = blocks.iter().copied().filter(|&z| loudness(z) > threshold);
        kept.collect()
    };
    let absolute = gated(-70.0);
    if absolute.is_empty() {
        return f64::NEG_INFINITY;
    }
    let relative = loudness(mean(&absolute)) - 10.0;
    loudness(mean(&gated(relative.max(-70.0))))
}

/// The true peak of `samples`, in dBTP: the largest magnitude of them oversampled four times by
/// the interpolating filter of BS.1770-4's Annex 2, and never under that of the samples
/// themselves, or `-inf` for silence.
pub fn true_peak(samples: &[f64]) -> f64 {
    let taps = INTERPOLATOR[0].len();
    let sample_peak = samples.iter().fold(0f64, |m, s| m.max(s.abs()));
    let mut peak = sample_peak;
    for n in 0..samples.len() + taps - 1 {
        for phase in &INTERPOLATOR {
            let y: f64 = (phase.iter().enumerate())
                .filter_map(|(k, h)| Some(h * samples.get(n.checked_sub(k)?)?))
                .sum();
            peak = peak.max(y.abs());
        }
    }
    20.0 * peak.log10()
}

/// The loudness of a mean square, in LUFS.
fn loudness(mean_square: f64) -> f64 {
    -0.691 + 10.0 * mean_square.log10()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_square(samples: &[f64]) -> f64 {
    samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64
}

/// `samples` through BS.1770-4's two-stage pre-filter at `rate`, its coefficients derived as
/// libebur128 derives them, which gives the standard's own at 48 kHz.
fn k_weighted(samples: &[f64], rate: u32) -> Vec<f64> {
    let [shelf, high_pass] = k_weighting(rate);
    biquad(&biquad(samples, shelf), high_pass)
}

/// The two stages of the pre-filter at `rate`, each as `[b0, b1, b2, a1, a2]`.
fn k_weighting(rate: u32) -> [[f64; 5]; 2] {
    let rate = f64::from(rate);
    let (f0, gain, q) = (1681.974450955533, 3.999843853973347, 0.7071752369554196);
    let k = (PI * f0 / rate).tan();
    let vh = 10f64.powf(gain / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let shelf = [
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ];
    let (f0, q) = (38.13547087602444, 0.5003270373238773);
    let k = (PI * f0 / rate).tan();
    let a0 = 1.0 + k / q + k * k;
    let high_pass = [
        1.0,
        -2.0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ];
    [shelf, high_pass]
}

/// `x` through the second-order filter `[b0, b1, b2, a1, a2]`.
fn biquad(x: &[f64], [b0, b1, b2, a1, a2]: [f64; 5]) -> Vec<f64> {
    let (mut z1, mut z2) = (0.0, 0.0);
    x.iter()
        .map(|&x| {
            let y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            y
        })
        .collect()
}

/// BS.1770-4's 48-tap filter for four times oversampling, a row of 12 taps per phase.
pub(crate) const INTERPOLATOR: [[f64; 12]; 4] = [
    [
        0.0017089843750,
        0.0109863281250,
        -0.0196533203125,
        0.0332031250000,
        -0.0594482421875,
        0.1373291015625,
        0.9721679687500,
        -0.1022949218750,
        0.0476074218750,
        -0.0266113281250,
        0.0148925781250,
        -0.0083007812500,
    ],
    [
        -0.0291748046875,
        0.0292968750000,
        -0.0517578125000,
        0.0891113281250,
        -0.1665039062500,
        0.4650878906250,
        0.7797851562500,
        -0.2003173828125,
        0.1015625000000,
        -0.0582275390625,
        0.0330810546875,
        -0.0189208984375,
    ],
    [
        -0.0189208984375,
        0.0330810546875,
        -0.0582275390625,
        0.1015625000000,
        -0.2003173828125,
        0.7797851562500,
        0.4650878906250,
        -0.1665039062500,
        0.0891113281250,
        -0.0517578125000,
        0.0292968750000,
        -0.0291748046875,
    ],
    [
        -0.0083007812500,
        0.0148925781250,
        -0.0266113281250,
        0.0476074218750,
        -0.1022949218750,
        0.9721679687500,
        0.1373291015625,
        -0.0594482421875,
        0.0332031250000,
        -0.0196533203125,
        0.0109863281250,
        0.0017089843750,
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    /// BS.1770-4's Tables 1 and 2, which the derivation gives to within their last digits.
    #[test]
    fn k_weighting_is_the_standards_at_48_khz() {
        let tables = [
            [
                1.53512485958697,
                -2.69169618940638,
                1.19839281085285,
                -1.69065929318241,
                0.73248077421585,
            ],
            [1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621],
        ];
        for (stage, table) in k_weighting(48000).iter().zip(tables) {
            for (c, t) in stage.iter().zip(table) {
                assert!((c - t).abs() < 1e-13, "{c} for {t}");
            }
        }
    }
}
