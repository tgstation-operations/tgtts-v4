use voxcpm_core::minicpm4::Config;

use crate::Error;

/// What loading refuses of a [`Config`]: a trait, the config being `voxcpm_core`'s.
pub(crate) trait Check {
    fn check(&self) -> Result<(), Error>;
}

impl Check for Config {
    fn check(&self) -> Result<(), Error> {
        let unsupported = |what: String| Err(Error::Unsupported(what));
        if self.use_mup {
            return unsupported("use_mup".into());
        }
        let (heads, kv_heads) = (self.num_attention_heads, self.num_key_value_heads);
        if kv_heads == 0 || heads % kv_heads != 0 {
            return unsupported(format!(
                "{heads} attention heads shared among {kv_heads} key-value heads"
            ));
        }
        if self.no_rope {
            return Ok(());
        }
        let rope = &self.rope_scaling;
        if rope.kind != "longrope" {
            return unsupported(format!("rope_scaling of type {:?}", rope.kind));
        }
        if self.max_position_embeddings != rope.original_max_position_embeddings {
            return unsupported(format!(
                "LongRoPE stretched from {} positions to {}",
                rope.original_max_position_embeddings, self.max_position_embeddings
            ));
        }
        let head_dim = self.head_dim();
        if head_dim % 2 != 0 || rope.short_factor.len() != head_dim / 2 {
            return unsupported(format!(
                "{} RoPE short factors for heads {head_dim} wide",
                rope.short_factor.len()
            ));
        }
        Ok(())
    }
}
