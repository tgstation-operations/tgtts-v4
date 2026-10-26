use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};

use crate::Config;

/// The cosines and sines of upstream's LongRoPE for every position, `[positions, head_dim]`.
///
/// They stay f32 whatever the model's dtype, as nanovllm-voxcpm keeps them; upstream's
/// `model.to(bfloat16)` rounds its copy to bf16.
#[derive(Clone, Debug)]
pub(crate) struct Rope {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl Rope {
    pub(crate) fn new(config: &Config, device: &Device) -> Self {
        let dim = config.head_dim();
        let positions = config.max_position_embeddings;
        // Each step in f32 and in upstream's order, so the tables round as its do. The
        // scaling factor upstream multiplies them by is exactly 1 at the native length,
        // the only one `Config::check` admits.
        let theta = config.rope_theta as f32;
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dim as f32));
        let factors = config.rope_scaling.short_factor.iter();
        let rates: Vec<_> = factors
            .zip(inv_freq)
            .map(|(&f, inv)| (1.0 / f as f32, inv))
            .collect();
        let mut cos = Vec::with_capacity(positions * dim);
        let mut sin = Vec::with_capacity(positions * dim);
        for t in 0..positions {
            let freqs: Vec<f32> = rates.iter().map(|(r, inv)| t as f32 * r * inv).collect();
            for _ in 0..2 {
                cos.extend(freqs.iter().map(|f| f.cos()));
                sin.extend(freqs.iter().map(|f| f.sin()));
            }
        }
        let table = |values| Tensor::from_data(TensorData::new(values, [positions, dim]), device);
        Self {
            cos: table(cos),
            sin: table(sin),
        }
    }

    /// The rotations of positions `start..start + len`, `[1, 1, len, head_dim]`.
    ///
    /// # Panics
    ///
    /// If that reaches past `max_position_embeddings`.
    pub(crate) fn at(&self, start: usize, len: usize) -> Rotation {
        let [positions, _] = self.cos.dims();
        assert!(
            start + len <= positions,
            "position {} is past the {positions} RoPE has rotations for",
            start + len - 1
        );
        let rows = |table: &Tensor<2>| table.clone().narrow(0, start, len).unsqueeze();
        Rotation {
            cos: rows(&self.cos),
            sin: rows(&self.sin),
        }
    }

    /// The rotations of positions `start..start + len` for each row's `start`,
    /// `[starts.len(), 1, len, head_dim]`.
    ///
    /// # Panics
    ///
    /// If that reaches past `max_position_embeddings`.
    pub(crate) fn rows(&self, starts: &[usize], len: usize) -> Rotation {
        let [positions, dim] = self.cos.dims();
        let last = starts.iter().max().map_or(0, |start| start + len);
        assert!(
            last <= positions,
            "position {} is past the {positions} RoPE has rotations for",
            last - 1
        );
        let at: Vec<i64> = starts
            .iter()
            .flat_map(|&start| (start..start + len).map(|t| t as i64))
            .collect();
        let at = Tensor::<1, Int>::from_data(at.as_slice(), &self.cos.device());
        let rows = |table: &Tensor<2>| {
            let rows = table.clone().select(0, at.clone());
            rows.reshape([starts.len(), 1, len, dim])
        };
        Rotation {
            cos: rows(&self.cos),
            sin: rows(&self.sin),
        }
    }
}

/// The rotations of a run of positions, for [`Rotation::apply`].
pub(crate) struct Rotation {
    cos: Tensor<4>,
    sin: Tensor<4>,
}

impl Rotation {
    /// Rotates queries or keys `[batch, heads, len, head_dim]` in f32, as upstream does.
    pub(crate) fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        let dtype = x.dtype();
        let x = x.cast(FloatDType::F32);
        let half = x.dims()[3] / 2;
        let rotated = Tensor::cat(
            vec![
                x.clone().narrow(3, half, half).neg(),
                x.clone().narrow(3, 0, half),
            ],
            3,
        );
        (x * self.cos.clone() + rotated * self.sin.clone()).cast(dtype)
    }
}
