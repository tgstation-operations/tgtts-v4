use std::{f32::consts::FRAC_PI_2, path::Path};

use burn::tensor::{Device, FloatDType, Tensor};

use crate::{Config, Error, LocalDit};

/// How a patch is sampled. The default is what the Python service asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    /// Euler steps from noise to the patch, upstream's `inference_timesteps`.
    pub timesteps: usize,
    /// The scale of the noise the flow starts from.
    pub temperature: f32,
    /// The classifier-free guidance scale.
    pub cfg_value: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            timesteps: 10,
            temperature: 1.0,
            cfg_value: 2.0,
        }
    }
}

/// The sampler of VoxCPM2's patches, upstream's `UnifiedCFM` at inference: Euler steps along
/// the flow the local DiT predicts, from noise at time 1 to a patch at time 0, with
/// classifier-free guidance as CFG-Zero* has it.
///
/// It integrates in f32 whatever dtype the DiT computes in.
#[derive(Debug)]
pub struct Cfm {
    dit: LocalDit,
}

impl Cfm {
    /// Loads the sampler's local DiT, as [`LocalDit::load`] does.
    ///
    /// # Errors
    ///
    /// As [`LocalDit::load`].
    pub fn load(
        config: &Config,
        path: &Path,
        dtype: FloatDType,
        device: &Device,
    ) -> Result<Self, Error> {
        Ok(Self {
            dit: LocalDit::load(config, path, dtype, device)?,
        })
    }

    /// Patches `[batch, feat_dim, patch_size]` sampled from standard normal `noise` of that
    /// shape, given the LMs' hidden states `mu` `[batch, 2 * dit hidden]` and the previous
    /// patches `cond` `[batch, feat_dim, patch_size]`.
    pub fn sample(
        &self,
        mu: Tensor<2>,
        cond: Tensor<3>,
        noise: Tensor<3>,
        sampling: &Sampling,
    ) -> Tensor<3> {
        let x = noise.cast(FloatDType::F32).mul_scalar(sampling.temperature);
        let t_span = schedule(sampling.timesteps);
        self.solve_euler(x, &t_span, mu, cond, sampling.cfg_value)
    }

    /// Patches sampled as [`sample`](Self::sample) samples them, each row of the batch as its
    /// own of `sampling` says.
    ///
    /// # Panics
    ///
    /// If `sampling` has another number of rows, or its rows take other numbers of steps.
    pub fn sample_rows(
        &self,
        mu: Tensor<2>,
        cond: Tensor<3>,
        noise: Tensor<3>,
        sampling: &[Sampling],
    ) -> Tensor<3> {
        let [batch, ..] = noise.dims();
        assert_eq!(
            sampling.len(),
            batch,
            "sampling for {} rows",
            sampling.len()
        );
        let timesteps = sampling.first().map_or(0, |row| row.timesteps);
        assert!(
            sampling.iter().all(|row| row.timesteps == timesteps),
            "rows sampled in different numbers of steps"
        );
        let device = noise.device();
        let per_row = |value: fn(&Sampling) -> f32| {
            let values: Vec<f32> = sampling.iter().map(value).collect();
            Tensor::<1>::from_floats(values.as_slice(), &device).reshape([batch, 1, 1])
        };
        let x = noise.cast(FloatDType::F32) * per_row(|row| row.temperature);
        let cfg = per_row(|row| row.cfg_value);
        self.solve(x, &schedule(timesteps), mu, cond, |v| v * cfg.clone())
    }

    /// Integrates from `x` along the times `t_span`, as upstream's `solve_euler` does with
    /// CFG-Zero*: the first steps, 4% of them but at least one, stay where they are, and the
    /// unconditional velocity, the DiT's with `mu` zeroed, is scaled to best match the
    /// conditional one before guiding away from it.
    ///
    /// # Panics
    ///
    /// If `t_span` has fewer than two times.
    pub fn solve_euler(
        &self,
        x: Tensor<3>,
        t_span: &[f32],
        mu: Tensor<2>,
        cond: Tensor<3>,
        cfg_value: f32,
    ) -> Tensor<3> {
        self.solve(x, t_span, mu, cond, |v| v.mul_scalar(cfg_value))
    }

    /// [`solve_euler`](Self::solve_euler), guiding by what `guide` scales the difference
    /// between the velocities to.
    fn solve(
        &self,
        x: Tensor<3>,
        t_span: &[f32],
        mu: Tensor<2>,
        cond: Tensor<3>,
        guide: impl Fn(Tensor<3>) -> Tensor<3>,
    ) -> Tensor<3> {
        let mut x = x.cast(FloatDType::F32);
        let [batch, channels, len] = x.dims();
        let device = x.device();
        let mu = Tensor::cat(vec![mu.clone(), mu.zeros_like()], 0);
        let cond = Tensor::cat(vec![cond.clone(), cond], 0);
        // The step length is the DiT's input only in mean mode, which Config refuses.
        let dt_in = Tensor::<1>::zeros([2 * batch], &device);
        let zero_init_steps = 1.max((t_span.len() as f64 * 0.04) as usize);
        let (mut t, mut dt) = (t_span[0], t_span[0] - t_span[1]);
        for step in 1..t_span.len() {
            if step > zero_init_steps {
                let x_in = Tensor::cat(vec![x.clone(), x.clone()], 0);
                let t_in = Tensor::full([2 * batch], t, &device);
                let velocity =
                    self.dit
                        .forward(x_in, mu.clone(), t_in, cond.clone(), dt_in.clone());
                let mut halves = velocity.cast(FloatDType::F32).chunk(2, 0).into_iter();
                let (positive, negative) = (halves.next().unwrap(), halves.next().unwrap());
                let flat = |v: Tensor<3>| v.reshape([batch, channels * len]);
                let dot = flat(positive.clone() * negative.clone()).sum_dim(1);
                let norm = flat(negative.clone().square()).sum_dim(1).add_scalar(1e-8);
                let scaled = negative * (dot / norm).reshape([batch, 1, 1]);
                let guided = scaled.clone() + guide(positive - scaled);
                x = x - guided.mul_scalar(dt);
            }
            t -= dt;
            if step < t_span.len() - 1 {
                dt = t - t_span[step + 1];
            }
        }
        x
    }
}

/// The `timesteps + 1` times upstream's sampler steps through: evenly spaced from 1 to 0, as
/// torch's `linspace` computes them in f32, then swayed towards 1 by `t + cos(pi t / 2) - 1`.
pub fn schedule(timesteps: usize) -> Vec<f32> {
    let steps = timesteps + 1;
    let step = (0.0 - 1.0) / timesteps as f32;
    (0..steps)
        .map(|i| {
            let t = if i < steps / 2 {
                1.0 + step * i as f32
            } else {
                0.0 - step * (steps - i - 1) as f32
            };
            t + ((FRAC_PI_2 * t).cos() - 1.0 + t)
        })
        .collect()
}
