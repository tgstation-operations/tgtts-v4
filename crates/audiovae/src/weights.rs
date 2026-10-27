use std::{collections::BTreeMap, path::PathBuf};

use burn::{
    module::Module,
    store::{
        FloatCastAdapter, ModuleSnapshot, ModuleStore, PytorchStore, bridge,
        burn_pack::{Error as PackError, Tensor as PackTensor},
    },
    tensor::{DType, FloatDType, TensorData},
};

use crate::Error;

/// The tensors of a checkpoint or a fixture under their state_dict names, each read from its
/// file only as it loads.
pub struct Weights {
    tensors: BTreeMap<String, PackTensor>,
}

impl Weights {
    /// Opens a VoxCPM2 `audiovae.pth`.
    ///
    /// # Errors
    ///
    /// If it cannot be read as a PyTorch checkpoint with a `state_dict`.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, Error> {
        // The store would otherwise renumber `decoder.sr_cond_model`, whose indices start at
        // 2, from 0.
        let mut store = PytorchStore::from_file(path)
            .with_top_level_key("state_dict")
            .map_indices_contiguous(false);
        Self::from_store(&mut store)
    }

    /// Every tensor in `store`, under its own name.
    ///
    /// # Errors
    ///
    /// If the store cannot list its tensors.
    pub fn from_store(store: &mut impl ModuleStore) -> Result<Self, Error> {
        let tensors = store
            .get_all_tensors()
            .map_err(|e| Error::Weights(e.to_string()))?;
        Ok(Self {
            tensors: tensors.clone(),
        })
    }

    /// The tensor `name`, flattened, as integers.
    pub(crate) fn ints(&self, name: &str) -> Result<Vec<i64>, Error> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Weights(format!("no tensor {name}")))?;
        let data = bridge::to_data(tensor).map_err(|e| Error::Weights(format!("{name}: {e}")))?;
        Ok(data.iter::<i64>().collect())
    }

    /// The parameter `name` as f32, in torch's layout, and its shape: a weight-normed
    /// convolution's `<conv>.weight` folded from its `weight_g` and `weight_v` as loading folds
    /// it, for a runtime other than burn to take the values a module would load.
    ///
    /// # Errors
    ///
    /// If there is no such parameter, or it cannot be read as floats.
    pub fn floats(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), Error> {
        let tensor = match self.tensors.get(name) {
            Some(tensor) => tensor.clone(),
            None => {
                let part = |suffix: &str| {
                    self.tensors
                        .get(&format!("{name}{suffix}"))
                        .ok_or_else(|| Error::Weights(format!("no tensor {name}")))
                };
                folded(name.to_owned(), part("_g")?, part("_v")?.clone())?
            }
        };
        let values = f32s(&tensor).map_err(|e| Error::Weights(e.to_string()))?;
        Ok((tensor.shape.to_vec(), values))
    }

    /// `module` with its parameters set from the tensors named `<prefix>.<parameter path>`,
    /// cast to `dtype`, and weight norm folded. Every tensor under the prefix has to be used
    /// but those under `skip`, and every parameter set.
    pub(crate) fn load<M: Module>(
        &self,
        prefix: &str,
        mut module: M,
        dtype: FloatDType,
        skip: &[&str],
    ) -> Result<M, Error> {
        let skipped = |name: &str| {
            skip.iter().any(|skip| {
                name.strip_prefix(skip)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
            })
        };
        let tensors = self
            .tensors
            .iter()
            .filter_map(|(name, tensor)| {
                let name = name.strip_prefix(prefix)?.strip_prefix('.')?;
                let mut tensor = tensor.clone();
                tensor.name = name.to_owned();
                (!skipped(name)).then(|| (tensor.name.clone(), tensor))
            })
            .collect();
        let tensors = fold_weight_norm(tensors)?;
        let cast = FloatCastAdapter::to(dtype.into());
        let result = module.apply(tensors, None, Some(Box::new(cast)), true);
        if result.errors.is_empty() && result.missing.is_empty() && result.unused.is_empty() {
            Ok(module)
        } else {
            Err(Error::Weights(format!("{prefix}: {result}")))
        }
    }
}

/// Replaces each weight-normed pair `<conv>.weight_g`, `<conv>.weight_v` by the
/// `<conv>.weight` it stands for, which torch's `weight_norm` recomputes on every call.
fn fold_weight_norm(mut tensors: BTreeMap<String, PackTensor>) -> Result<Vec<PackTensor>, Error> {
    let pairs: Vec<String> = tensors
        .keys()
        .filter_map(|name| name.strip_suffix("_v"))
        .filter(|weight| parameter(weight) == "weight")
        .map(str::to_owned)
        .collect();
    for weight in pairs {
        let v = tensors
            .remove(&format!("{weight}_v"))
            .expect("listed above");
        let g = tensors
            .remove(&format!("{weight}_g"))
            .ok_or_else(|| Error::Weights(format!("{weight}_v has no {weight}_g")))?;
        tensors.insert(weight.clone(), folded(weight, &g, v)?);
    }
    if let Some(g) = tensors.keys().find(|name| parameter(name) == "weight_g") {
        return Err(Error::Weights(format!("{g} has no weight_v")));
    }
    Ok(tensors.into_values().collect())
}

fn parameter(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// `g * v / |v|`, with the norm taken over every axis of `v` but the first, as torch's
/// `_weight_norm(v, g, 0)` computes it: in f32, from a norm accumulated here in f64.
fn folded(name: String, g: &PackTensor, v: PackTensor) -> Result<PackTensor, Error> {
    let shape = v.shape.clone();
    let fits = g.shape.len() == shape.len()
        && g.shape.first() == shape.first()
        && g.shape.iter().skip(1).all(|&dim| dim == 1);
    if !fits {
        return Err(Error::Weights(format!(
            "{name}_g of shape {:?} does not fit {name}_v of shape {shape:?}",
            g.shape
        )));
    }
    let g = g.clone();
    let folded_shape = shape.clone();
    Ok(bridge::deferred(name, DType::F32, shape, None, move || {
        let (g, v) = (f32s(&g)?, f32s(&v)?);
        let row = v.len() / g.len();
        let weight: Vec<f32> = v
            .chunks(row)
            .zip(g)
            .flat_map(|(v, g)| {
                let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
                let scale = g / norm as f32;
                v.iter().map(move |&x| x * scale)
            })
            .collect();
        Ok(TensorData::new(weight, folded_shape.clone()))
    }))
}

fn f32s(tensor: &PackTensor) -> Result<Vec<f32>, PackError> {
    bridge::to_data(tensor)?
        .try_to_vec_as::<f32>()
        .map_err(|e| PackError::ValidationError(format!("{}: {e:?}", tensor.name)))
}

#[cfg(test)]
mod tests {
    use burn::{nn::conv::Conv1dConfig, store::SafetensorsStore};
    use testkit::{device, fixture};

    use super::*;
    use crate::layers::causal;

    #[test]
    fn floats_are_what_a_module_loads() {
        let mut store = SafetensorsStore::from_file(fixture!("audiovae").path());
        let weights = Weights::from_store(&mut store).unwrap();
        let conv = causal(Conv1dConfig::new(3, 4, 10).with_stride(5), 3, 1).init(&device());
        let conv = weights.load("conv", conv, FloatDType::F32, &[]).unwrap();
        let loaded = |data: TensorData| data.try_to_vec::<f32>().unwrap();
        let (shape, folded) = weights.floats("conv.weight").unwrap();
        assert_eq!(shape, [4, 3, 10]);
        assert_eq!(folded, loaded(conv.weight.val().into_data()));
        let bias = conv.bias.as_ref().unwrap().val().into_data();
        assert_eq!(
            weights.floats("conv.bias").unwrap(),
            (vec![4], loaded(bias))
        );
        assert!(weights.floats("conv.scale").is_err());
    }
}
