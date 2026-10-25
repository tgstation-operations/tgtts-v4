//! The benchmarks, at the shapes a request runs them at.

use std::{fs, path::Path};

use audiovae::{AudioVaeConfig, Decoder, Encoder, Weights};
use burn::module::Module;
use minicpm4::{Config, KvCache, LocalShape, Mask, Model};
use serde_json::Value;
use voices::Voices;

use crate::{
    conditions::BIGMEM_LOCK,
    runner::{Runner, Spec},
};

pub const NAMES: &[&str] = &[
    "minicpm4.base_lm.load",
    "minicpm4.base_lm.prefill",
    "minicpm4.base_lm.step",
    "minicpm4.residual_lm.load",
    "minicpm4.residual_lm.prefill",
    "minicpm4.residual_lm.step",
    "minicpm4.local_encoder.load",
    "minicpm4.local_encoder.patch",
    "minicpm4.local_encoder.prompt",
    "minicpm4.local_dit.load",
    "minicpm4.local_dit.eval",
    "minicpm4.patch",
    "audiovae.decoder.load",
    "audiovae.decode.utterance",
    "audiovae.decode.patch",
    "audiovae.encoder.load",
    "audiovae.encode",
    "voices.open",
    "voices.latent",
    "voices.blips.open",
    "voices.blips.sample",
];

/// The positions of the prompt prod builds for the median voice of the converted set: the
/// reference prefix (179 patches between two markers), a 105-token transcript, a 15-token
/// line, the audio start, and the 179 patches again as the continuation.
const PROMPT: usize = 481;
/// The patches prod generates at most, which the KV caches hold beyond the prompt.
const MAX_PATCHES: usize = 256;
const FRAMES_PER_PATCH: usize = 4;
/// A latent frame is 40 ms of audio.
const FRAME_SECONDS: f64 = 0.04;
/// Of the 10 Euler steps, CFG-Zero* zeroes the first.
const DIT_EVALS: usize = 9;
/// Tokens the local encoder takes per patch: a CLS token and the patch's frames.
const ENCODER_TOKENS: usize = 1 + FRAMES_PER_PATCH;
/// Tokens the local DiT takes: mu from both LMs, the time, a patch of condition and one of x.
const DIT_TOKENS: usize = 3 + 2 * FRAMES_PER_PATCH;
/// About the patches a 15-token line comes to.
const UTTERANCE_PATCHES: usize = 25;
/// Prod decodes each new patch with the 12 frames before it as context.
const DECODE_CONTEXT_FRAMES: usize = 12;
const ENCODE_SECONDS: usize = 10;

/// Where a run reads the real weights and voices from.
pub struct Data<'a> {
    pub checkpoint: Option<&'a Path>,
    pub voices: Option<&'a Path>,
    pub voice: Option<&'a str>,
}

/// Refuses, before anything is timed, a selection that could not run to the end.
pub fn check(r: &Runner, data: &Data) -> Result<(), String> {
    if !NAMES.iter().any(|name| r.selected(name)) {
        return Err("no benchmark matches the filters".into());
    }
    let mut missing = Vec::new();
    let needs_checkpoint = r.any_selected("minicpm4.") || r.any_selected("audiovae.");
    if needs_checkpoint && data.checkpoint.is_none() {
        missing.push("VOXCPM2_DIR to name the VoxCPM2 checkpoint");
    }
    if r.any_selected("voices.") && data.voices.is_none() {
        missing.push("VOICES_DIR to name a converted voices directory");
    }
    if !missing.is_empty() {
        return Err(format!(
            "the selected benchmarks need {}",
            missing.join(" and ")
        ));
    }
    let steps = [step_spec("base_lm"), patch_spec()]
        .iter()
        .map(|spec| r.counts(spec))
        .map(|(warmup, reps)| warmup + reps)
        .sum::<usize>();
    if steps > MAX_PATCHES {
        return Err(format!(
            "the step and patch benchmarks would take {steps} decode steps, more than the \
             {MAX_PATCHES} the KV caches hold past the prompt"
        ));
    }
    Ok(())
}

/// Runs every selected benchmark, a group at a time.
pub fn run(r: &mut Runner, data: &Data) -> Result<(), String> {
    if r.any_selected("minicpm4.") {
        minicpm4(r, data.checkpoint.expect("checked"))?;
    }
    if r.any_selected("audiovae.") {
        audiovae(r, data.checkpoint.expect("checked"))?;
    }
    if r.any_selected("voices.") {
        voices(r, data.voices.expect("checked"), data.voice)?;
    }
    Ok(())
}

fn minicpm4(r: &mut Runner, dir: &Path) -> Result<(), String> {
    let json = config_json(dir)?;
    let lm: Config = serde_json::from_value(json["lm_config"].clone())
        .map_err(|e| format!("config.json: lm_config: {e}"))?;
    let no_rope = json["residual_lm_no_rope"].as_bool();
    let residual = lm.residual_lm(
        count(&json, "residual_lm_num_layers")?,
        no_rope.ok_or("config.json: residual_lm_no_rope is not a boolean")?,
    );
    let encoder = lm.local(&local_shape(&json["encoder_config"])?);
    let dit = lm.local(&local_shape(&json["dit_config"])?);
    let weights = dir.join("model.safetensors");
    let _bigmem = r.lock(BIGMEM_LOCK)?;

    let wanted = |r: &Runner, stack: &str| {
        r.any_selected(&format!("minicpm4.{stack}.")) || r.selected("minicpm4.patch")
    };
    let stepped = |r: &Runner, stack: &str| {
        let [prefill, step] = ["prefill", "step"].map(|b| format!("minicpm4.{stack}.{b}"));
        r.selected(&prefill) || r.selected(&step) || r.selected("minicpm4.patch")
    };
    let load = |r: &mut Runner, stack: &str, config: &Config, prefix: &str| {
        load_stack(r, stack, config, &weights, prefix)
    };

    let base = wanted(r, "base_lm").then(|| load(r, "base_lm", &lm, "base_lm."));
    let base_cache = base
        .as_ref()
        .filter(|_| stepped(r, "base_lm"))
        .map(|model| lm_benches(r, "base_lm", model, lm.hidden_size, 1));

    let residual_lm =
        wanted(r, "residual_lm").then(|| load(r, "residual_lm", &residual, "residual_lm."));
    let residual_cache = residual_lm
        .as_ref()
        .filter(|_| stepped(r, "residual_lm"))
        .map(|model| lm_benches(r, "residual_lm", model, residual.hidden_size, 3));

    let local_encoder = wanted(r, "local_encoder")
        .then(|| load(r, "local_encoder", &encoder, "feat_encoder.encoder."));
    let encoder_patch = r.seeded([1, ENCODER_TOKENS, encoder.hidden_size], 5);
    if let Some(model) = &local_encoder {
        let spec = Spec::new("minicpm4.local_encoder.patch", 5, 30)
            .work(1.0, "patches")
            .param("batch", 1)
            .param("len", ENCODER_TOKENS);
        r.measure(
            spec,
            || (),
            |()| model.forward(encoder_patch.clone(), Mask::Bidirectional),
        );
        let prompt = r.seeded([PROMPT, ENCODER_TOKENS, encoder.hidden_size], 6);
        let spec = Spec::new("minicpm4.local_encoder.prompt", 1, 5)
            .work(PROMPT as f64, "patches")
            .param("batch", PROMPT)
            .param("len", ENCODER_TOKENS);
        r.measure(
            spec,
            || (),
            |()| model.forward(prompt.clone(), Mask::Bidirectional),
        );
    }

    let local_dit = wanted(r, "local_dit")
        .then(|| load(r, "local_dit", &dit, "feat_decoder.estimator.decoder."));
    // A batch of two: the conditioned and the unconditioned halves of classifier-free guidance.
    let dit_input = r.seeded([2, DIT_TOKENS, dit.hidden_size], 7);
    if let Some(model) = &local_dit {
        let spec = Spec::new("minicpm4.local_dit.eval", 5, 30)
            .work(1.0, "evals")
            .param("batch", 2)
            .param("len", DIT_TOKENS);
        r.measure(
            spec,
            || (),
            |()| model.forward(dit_input.clone(), Mask::Bidirectional),
        );
    }

    // A patch's worth of the four stacks as generation runs them, each depending on the one
    // before as there, but without the glue between them: the projections, FSQ, the stop head
    // and the Euler arithmetic.
    if r.selected("minicpm4.patch") {
        let loaded = "every stack is loaded for the patch";
        let (base, mut base_cache) = (base.expect(loaded), base_cache.expect(loaded));
        let residual_lm = residual_lm.expect(loaded);
        let mut residual_cache = residual_cache.expect(loaded);
        let (local_encoder, local_dit) = (local_encoder.expect(loaded), local_dit.expect(loaded));
        let base_input = r.seeded([1, 1, lm.hidden_size], 8);
        let (warmup, reps) = r.counts(&step_spec("base_lm"));
        let stepped = if r.selected("minicpm4.base_lm.step") {
            warmup + reps
        } else {
            0
        };
        let spec = patch_spec()
            .work(1.0, "patches")
            .audio(FRAMES_PER_PATCH as f64 * FRAME_SECONDS)
            .param("context_from", PROMPT + stepped)
            .param("dit_evals", DIT_EVALS)
            .param("stacks_only", true);
        r.measure(
            spec,
            || (),
            |()| {
                let hidden = base.forward_cached(base_input.clone(), &mut base_cache);
                let residual = residual_lm.forward_cached(hidden, &mut residual_cache);
                let embedded = local_encoder.forward(encoder_patch.clone(), Mask::Bidirectional);
                let velocity = (0..DIT_EVALS).fold(dit_input.clone(), |x, _| {
                    local_dit.forward(x, Mask::Bidirectional)
                });
                (residual, embedded, velocity)
            },
        );
    }
    Ok(())
}

fn load_stack(r: &mut Runner, stack: &str, config: &Config, weights: &Path, prefix: &str) -> Model {
    let (dtype, device) = (r.dtype, r.device.clone());
    let load = || {
        Model::load(config, weights, prefix, dtype, &device)
            .unwrap_or_else(|e| panic!("{prefix}: {e}"))
    };
    let name = format!("minicpm4.{stack}.load");
    let spec = Spec::new(&name, 1, 3)
        .param("layers", config.num_hidden_layers)
        .param("page_cache", "warm");
    let model = r.measure(spec, || (), |()| load());
    let model = model.unwrap_or_else(load);
    r.set_work(&name, model.num_params() as f64, "parameters");
    model
}

fn step_spec(stack: &str) -> Spec {
    Spec::new(&format!("minicpm4.{stack}.step"), 5, 30)
}

fn patch_spec() -> Spec {
    Spec::new("minicpm4.patch", 3, 20)
}

/// A causal stack's prefill of the prompt, then steps from its end; gives the cache the steps
/// left, for the patch to go on from.
fn lm_benches(r: &mut Runner, stack: &str, model: &Model, hidden: usize, seed: u64) -> KvCache {
    let prompt = r.seeded([1, PROMPT, hidden], seed);
    let empty = || model.cache(1, PROMPT + MAX_PATCHES);
    let prefill = |mut cache: KvCache| {
        let out = model.forward_cached(prompt.clone(), &mut cache);
        (cache, out)
    };
    let spec = Spec::new(&format!("minicpm4.{stack}.prefill"), 1, 5)
        .work(PROMPT as f64, "tokens")
        .param("batch", 1)
        .param("len", PROMPT);
    let filled = r.measure(spec, empty, prefill);
    let (mut cache, _) = filled.unwrap_or_else(|| prefill(empty()));

    let step = r.seeded([1, 1, hidden], seed + 1);
    let spec = step_spec(stack)
        .work(1.0, "tokens")
        .param("context_from", PROMPT)
        .param("capacity", PROMPT + MAX_PATCHES);
    r.measure(
        spec,
        || (),
        |()| model.forward_cached(step.clone(), &mut cache),
    );
    cache
}

fn audiovae(r: &mut Runner, dir: &Path) -> Result<(), String> {
    let json = config_json(dir)?;
    let config: AudioVaeConfig = serde_json::from_value(json["audio_vae_config"].clone())
        .map_err(|e| format!("config.json: audio_vae_config: {e}"))?;
    let path = dir.join("audiovae.pth");
    let (dtype, device) = (r.dtype, r.device.clone());
    let weights = || Weights::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let frame_seconds =
        config.decoder_rates.iter().product::<usize>() as f64 / f64::from(config.out_sample_rate);

    if r.any_selected("audiovae.decode") {
        let load = || {
            Decoder::load(&config, &weights(), dtype, &device)
                .unwrap_or_else(|e| panic!("decoder: {e}"))
        };
        let spec = Spec::new("audiovae.decoder.load", 1, 5).param("page_cache", "warm");
        let decoder = r.measure(spec, || (), |()| load()).unwrap_or_else(load);

        let frames = UTTERANCE_PATCHES * FRAMES_PER_PATCH;
        let latents = r.seeded([1, config.latent_dim, frames], 9);
        let spec = Spec::new("audiovae.decode.utterance", 1, 10)
            .audio(frames as f64 * frame_seconds)
            .param("frames", frames);
        r.measure(spec, || (), |()| decoder.decode(latents.clone()));

        let frames = DECODE_CONTEXT_FRAMES + FRAMES_PER_PATCH;
        let latents = r.seeded([1, config.latent_dim, frames], 10);
        let spec = Spec::new("audiovae.decode.patch", 3, 30)
            .audio(FRAMES_PER_PATCH as f64 * frame_seconds)
            .param("frames", frames)
            .param("new_frames", FRAMES_PER_PATCH);
        r.measure(spec, || (), |()| decoder.decode(latents.clone()));
    }

    if r.any_selected("audiovae.encode") {
        let load = || {
            Encoder::load(&config, &weights(), dtype, &device)
                .unwrap_or_else(|e| panic!("encoder: {e}"))
        };
        let spec = Spec::new("audiovae.encoder.load", 1, 5).param("page_cache", "warm");
        let encoder = r.measure(spec, || (), |()| load()).unwrap_or_else(load);

        let samples = ENCODE_SECONDS * config.sample_rate as usize;
        let audio = r.seeded([1, 1, samples], 11);
        let spec = Spec::new("audiovae.encode", 1, 5)
            .audio(ENCODE_SECONDS as f64)
            .param("samples", samples);
        r.measure(spec, || (), |()| encoder.encode(audio.clone()));
    }
    Ok(())
}

fn voices(r: &mut Runner, dir: &Path, voice: Option<&str>) -> Result<(), String> {
    let open = || Voices::open(dir).map_err(|e| e.to_string());
    let table = open()?;
    let voice = match voice {
        Some(voice) => voice.to_owned(),
        None => table
            .names()
            .next()
            .ok_or("the voices table is empty")?
            .to_owned(),
    };
    let latent = table.latent(&voice).map_err(|e| e.to_string())?;
    let blips = table.blips(&voice).map_err(|e| e.to_string())?;
    let sample = ["male", "1", "a"];
    blips
        .sample(sample[0], sample[1], sample[2])
        .map_err(|e| e.to_string())?;

    let spec = Spec::new("voices.open", 10, 100).param("voices", table.names().count());
    r.measure(spec, || (), |()| open().expect("opened before"));
    let spec = Spec::new("voices.latent", 10, 100)
        .param("voice", voice.as_str())
        .param("frames", latent.frames.len())
        .param("page_cache", "warm");
    r.measure(spec, || (), |()| table.latent(&voice).expect("read before"));
    let spec = Spec::new("voices.blips.open", 10, 100)
        .param("voice", voice.as_str())
        .param("page_cache", "warm");
    r.measure(
        spec,
        || (),
        |()| table.blips(&voice).expect("opened before"),
    );
    let spec = Spec::new("voices.blips.sample", 10, 100)
        .param("voice", voice.as_str())
        .param("sample", sample.join("/"))
        .param("page_cache", "warm");
    r.measure(
        spec,
        || (),
        |()| {
            blips
                .sample(sample[0], sample[1], sample[2])
                .expect("read before")
        },
    );
    Ok(())
}

fn config_json(dir: &Path) -> Result<Value, String> {
    let path = dir.join("config.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn count(json: &Value, key: &str) -> Result<usize, String> {
    let value = json[key].as_u64();
    value
        .map(|n| n as usize)
        .ok_or_else(|| format!("config.json: {key} is not a count"))
}

fn local_shape(json: &Value) -> Result<LocalShape, String> {
    Ok(LocalShape {
        hidden_dim: count(json, "hidden_dim")?,
        ffn_dim: count(json, "ffn_dim")?,
        num_heads: count(json, "num_heads")?,
        num_layers: count(json, "num_layers")?,
        kv_channels: json["kv_channels"].as_u64().map(|n| n as usize),
    })
}
