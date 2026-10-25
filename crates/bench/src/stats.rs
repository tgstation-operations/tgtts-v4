use serde::{Deserialize, Serialize};

/// Order statistics of a benchmark's timings, in seconds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

impl Summary {
    /// # Panics
    ///
    /// If `samples` is empty or holds a NaN.
    pub fn of(samples: &[f64]) -> Self {
        assert!(!samples.is_empty(), "no samples to summarize");
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("timings are not NaN"));
        let (q1, q3) = (quantile(&sorted, 0.25), quantile(&sorted, 0.75));
        Self {
            median: quantile(&sorted, 0.5),
            q1,
            q3,
            iqr: q3 - q1,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        }
    }
}

/// The `q` quantile of ascending `sorted`, interpolated linearly between the nearest ranks:
/// numpy's default, R's type 7.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let rank = q * (sorted.len() - 1) as f64;
    let (below, above) = (rank.floor() as usize, rank.ceil() as usize);
    sorted[below] + (sorted[above] - sorted[below]) * (rank - below as f64)
}

/// A duration in seconds with a unit that keeps three or four significant digits in view.
pub fn duration(seconds: f64) -> String {
    let (scale, unit) = match seconds.abs() {
        s if s >= 1.0 => (1.0, "s"),
        s if s >= 1e-3 => (1e3, "ms"),
        _ => (1e6, "us"),
    };
    format!("{:.3} {unit}", seconds * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quartiles_interpolate_between_ranks() {
        let summary = Summary::of(&[4.0, 1.0, 3.0, 2.0]);
        assert_eq!((summary.q1, summary.median, summary.q3), (1.75, 2.5, 3.25));
        assert_eq!((summary.min, summary.max, summary.iqr), (1.0, 4.0, 1.5));
    }

    #[test]
    fn one_sample_is_every_statistic() {
        let summary = Summary::of(&[0.5]);
        assert_eq!(
            [summary.q1, summary.median, summary.q3, summary.mean],
            [0.5; 4]
        );
        assert_eq!(summary.iqr, 0.0);
    }

    #[test]
    fn durations_pick_a_unit() {
        assert_eq!(duration(12.3456), "12.346 s");
        assert_eq!(duration(0.0123456), "12.346 ms");
        assert_eq!(duration(0.0000123), "12.300 us");
    }
}
