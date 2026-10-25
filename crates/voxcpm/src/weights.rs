use std::path::Path;

use burn::{
    module::Module,
    store::{
        FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    },
    tensor::FloatDType,
};

use crate::Error;

/// `module` with every parameter set from the tensor at `prefix` and its path in the
/// safetensors file at `path`, cast to `dtype`. Tensors outside the prefix, and those under it
/// that the module has no parameter for, are not read.
pub(crate) fn load<M: Module>(
    mut module: M,
    path: &Path,
    prefix: &str,
    dtype: FloatDType,
) -> Result<M, Error> {
    let adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype.into()));
    let mut store = SafetensorsStore::from_file(path).with_from_adapter(adapter);
    if !prefix.is_empty() {
        store = store.with_key_remapping(format!("^{}", regex::escape(prefix)), "");
    }
    module.load_from(&mut store).map_err(Error::Load)?;
    Ok(module)
}
