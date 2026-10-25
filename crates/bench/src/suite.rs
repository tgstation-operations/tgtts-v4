//! The benchmarks, at the shapes a request runs them at.

use std::{fs, path::Path};

use audiovae::{AudioVaeConfig, Decoder, Encoder, Weights};
use burn::module::Module;
use dsp::radio::{Assets, Pcm};
use minicpm4::{Config, KvCache, LocalShape, Mask, Model};
use serde_json::Value;
use voices::{Latent, Voices};
use voxcpm::{Dtypes, Params, SeededNoise, TextToSpeech};

use crate::{
    conditions::BIGMEM_LOCK,
    runner::{Runner, Spec, SplitMix, uniform},
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
    "voxcpm.load",
    "voxcpm.prefill",
    "voxcpm.patch",
    "voxcpm.synthesize",
    "audiovae.decoder.load",
    "audiovae.decode.utterance",
    "audiovae.decode.patch",
    "audiovae.encoder.load",
    "audiovae.encode",
    "voices.open",
    "voices.latent",
    "voices.blips.open",
    "voices.blips.sample",
    "dsp.speech.finish",
    "dsp.radio.assets",
    "dsp.radio",
    "dsp.gibberish",
    "dsp.blips.message",
    "dsp.sentences.segment",
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
/// The line the prompt's 15 tokens stand for.
const LINE: &str = "Hello there, could someone please come to medbay? I need help.";
/// The rate the model's speech is at.
const SPEECH_RATE: u32 = 48000;
/// The quarter tones a voice is shifted by; any but 0 costs the same.
const PITCH: i32 = 2;
/// Spans of an utterance the gibberish radio ducks, in seconds.
const GIBBERISH_SPANS: [(f64, f64); 3] = [(0.5, 0.9), (1.6, 2.2), (3.0, 3.4)];

/// Where a run reads the real weights, voices and sounds from.
pub struct Data<'a> {
    pub checkpoint: Option<&'a Path>,
    pub voices: Option<&'a Path>,
    pub voice: Option<&'a str>,
    pub assets: Option<&'a Path>,
}

/// Refuses, before anything is timed, a selection that could not run to the end.
pub fn check(r: &Runner, data: &Data) -> Result<(), String> {
    if !NAMES.iter().any(|name| r.selected(name)) {
        return Err("no benchmark matches the filters".into());
    }
    let mut missing = Vec::new();
    let needs_checkpoint =
        r.any_selected("minicpm4.") || r.any_selected("voxcpm.") || r.any_selected("audiovae.");
    if needs_checkpoint && data.checkpoint.is_none() {
        missing.push("VOXCPM2_DIR to name the VoxCPM2 checkpoint");
    }
    let needs_voices =
        r.any_selected("voices.") || r.any_selected("voxcpm.") || r.any_selected("dsp.blips");
    if needs_voices && data.voices.is_none() {
        missing.push("VOICES_DIR to name a converted voices directory");
    }
    let needs_assets = r.any_selected("dsp.radio") || r.any_selected("dsp.gibberish");
    if needs_assets && data.assets.is_none() {
        missing.push("ASSETS_DIR to name the directory of the radio's sounds");
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
    let (warmup, reps) = r.counts(&patch_step_spec());
    let steps = steps.max(warmup + reps);
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
    if r.any_selected("voxcpm.") {
        voxcpm(r, data)?;
    }
    if r.any_selected("audiovae.") {
        audiovae(r, data.checkpoint.expect("checked"))?;
    }
    if r.any_selected("voices.") {
        voices(r, data.voices.expect("checked"), data.voice)?;
    }
    if r.any_selected("dsp.") {
        dsp(r, data)?;
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

/// The voice whose latent has the median length, as the prompt's length sets the prefill's
/// cost: of the table's voices sorted by frames, then name.
fn median_voice(table: &Voices) -> Result<(String, Latent), String> {
    let mut lengths = Vec::new();
    for name in table.names() {
        let latent = table.latent(name).map_err(|e| e.to_string())?;
        lengths.push((latent.frames.len(), name.to_owned()));
    }
    lengths.sort();
    let (_, name) = lengths
        .get(lengths.len() / 2)
        .ok_or("the voices table is empty")?;
    let latent = table.latent(name).map_err(|e| e.to_string())?;
    Ok((name.clone(), latent))
}

fn voxcpm(r: &mut Runner, data: &Data) -> Result<(), String> {
    let dir = data.checkpoint.expect("checked");
    let table = Voices::open(data.voices.expect("checked")).map_err(|e| e.to_string())?;
    let (voice, latent) = match data.voice {
        Some(voice) => {
            let latent = table.latent(voice).map_err(|e| e.to_string())?;
            (voice.to_owned(), latent)
        }
        None => median_voice(&table)?,
    };
    let dtype = r.dtype;
    let dtypes = Dtypes {
        lm: dtype,
        local_encoder: dtype,
        local_dit: dtype,
        audio_decoder: dtype,
    };
    let params = Params::default();
    let device = r.device.clone();
    let _bigmem = r.lock(BIGMEM_LOCK)?;

    let load =
        || TextToSpeech::load(dir, &dtypes, &device).unwrap_or_else(|e| panic!("VoxCPM2: {e}"));
    let spec = Spec::new("voxcpm.load", 1, 2).param("page_cache", "warm");
    let tts = r.measure(spec, || (), |()| load()).unwrap_or_else(load);
    let prompt = tts.prompt(LINE, &latent).map_err(|e| e.to_string())?;
    let model = tts.model();

    if r.selected("voxcpm.prefill") || r.selected("voxcpm.patch") {
        let start = || {
            model
                .start(&prompt, params.max_patches)
                .expect("the prompt fits the LM")
        };
        let spec = Spec::new("voxcpm.prefill", 1, 3)
            .work(prompt.len() as f64, "positions")
            .param("voice", voice.as_str())
            .param("positions", prompt.len());
        let generation = r.measure(spec, || (), |()| start());
        let mut generation = generation.unwrap_or_else(start);
        let mut noise = SeededNoise::new(14);
        let spec = patch_step_spec()
            .work(1.0, "patches")
            .audio(FRAMES_PER_PATCH as f64 * FRAME_SECONDS)
            .param("timesteps", params.sampling.timesteps);
        r.measure(
            spec,
            || model.noise(&mut noise),
            |noise| {
                let step = generation.sample(noise, &params.sampling);
                generation.advance(step.patch.clone());
                step
            },
        );
    }

    let spec = Spec::new("voxcpm.synthesize", 0, 2)
        .param("voice", voice.as_str())
        .param("text", LINE)
        .param("max_patches", params.max_patches);
    let audio = r.measure(
        spec,
        || SeededNoise::new(15),
        |mut noise| {
            tts.synthesize(LINE, &latent, &params, &mut noise)
                .expect("the prompt fits the LM")
        },
    );
    if let Some(audio) = audio {
        let seconds = audio.len() as f64 / f64::from(tts.out_sample_rate());
        r.set_audio("voxcpm.synthesize", seconds);
    }
    Ok(())
}

fn patch_step_spec() -> Spec {
    Spec::new("voxcpm.patch", 2, 10)
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

/// `voice`, or the first voice of the table.
fn pick(table: &Voices, voice: Option<&str>) -> Result<String, String> {
    match voice {
        Some(voice) => Ok(voice.to_owned()),
        None => Ok(table
            .names()
            .next()
            .ok_or("the voices table is empty")?
            .to_owned()),
    }
}

fn voices(r: &mut Runner, dir: &Path, voice: Option<&str>) -> Result<(), String> {
    let open = || Voices::open(dir).map_err(|e| e.to_string());
    let table = open()?;
    let voice = pick(&table, voice)?;
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

impl dsp::Random for SplitMix {
    fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.random() * n as f64) as usize
    }
}

fn dsp(r: &mut Runner, data: &Data) -> Result<(), String> {
    let seconds = (UTTERANCE_PATCHES * FRAMES_PER_PATCH) as f64 * FRAME_SECONDS;
    let samples = uniform(12, (seconds * f64::from(SPEECH_RATE)) as usize);
    let speech = Pcm {
        rate: SPEECH_RATE,
        channels: 1,
        samples: samples.iter().map(|&s| (s * 8000.0) as i16).collect(),
    };

    if r.selected("dsp.speech.finish") {
        let segment = dsp::segment::Segment::new(speech.rate, speech.samples.clone());
        let spec = Spec::new("dsp.speech.finish", 2, 20)
            .audio(seconds)
            .param("pitch", PITCH);
        r.measure(spec, || (), |()| dsp::speech::finish(&segment, PITCH));
    }

    if r.any_selected("dsp.radio") || r.any_selected("dsp.gibberish") {
        let dir = data.assets.expect("checked");
        let [on, off, noise] =
            ["mic_click_on.wav", "mic_click_off.wav", "diffstatic.wav"].map(|f| dir.join(f));
        let (on, off, noise) = (read_wav(&on)?, read_wav(&off)?, read_wav(&noise)?);
        let match_assets = || Assets::new(&on, &off, &noise);
        let spec = Spec::new("dsp.radio.assets", 1, 5).param("rate", dsp::RATE);
        let assets = r.measure(spec, || (), |()| match_assets());
        let assets = assets.unwrap_or_else(match_assets);
        let spec = Spec::new("dsp.radio", 2, 20).audio(seconds);
        r.measure(spec, || (), |()| dsp::radio::radio(&speech, &assets));
        let spec = Spec::new("dsp.gibberish", 2, 20)
            .audio(seconds)
            .param("spans", GIBBERISH_SPANS.len());
        r.measure(
            spec,
            || (),
            |()| {
                dsp::gibberish::gibberish(&speech, &assets, &GIBBERISH_SPANS)
                    .expect("spans within the speech")
            },
        );
    }

    if r.selected("dsp.blips.message") {
        let table = Voices::open(data.voices.expect("checked")).map_err(|e| e.to_string())?;
        let voice = pick(&table, data.voice)?;
        let blips = table.blips(&voice).map_err(|e| e.to_string())?;
        let mut draws = SplitMix(13);
        let spec = Spec::new("dsp.blips.message", 3, 30)
            .param("voice", voice.as_str())
            .param("text", LINE);
        r.measure(
            spec,
            || (),
            |()| {
                dsp::blips::message(LINE, &blips, "male", "1", 0, &mut draws)
                    .expect("a known voice")
            },
        );
    }

    if r.selected("dsp.sentences.segment") {
        let spec = Spec::new("dsp.sentences.segment", 10, 100).param("text", LINE);
        r.measure(spec, || (), |()| dsp::sentences::segment(LINE));
    }
    Ok(())
}

/// A WAV file of 16-bit PCM.
fn read_wav(path: &Path) -> Result<Pcm, String> {
    let invalid = |why: &str| format!("{}: {why}", path.display());
    let bytes = fs::read(path).map_err(|e| invalid(&e.to_string()))?;
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(invalid("not a RIFF WAVE file"));
    }
    let (mut format, mut samples, mut at) = (None, None, 12);
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("4 bytes")) as usize;
        let body = bytes
            .get(at + 8..at + 8 + size)
            .ok_or_else(|| invalid("a chunk runs past the end"))?;
        let u16_at = |i: usize| u16::from_le_bytes([body[i], body[i + 1]]);
        match id {
            b"fmt " if size >= 16 => {
                let rate = u32::from_le_bytes(body[4..8].try_into().expect("4 bytes"));
                format = Some((u16_at(0), u16_at(2), rate, u16_at(14)));
            }
            b"data" => {
                let pcm = body.as_chunks().0.iter().map(|&s| i16::from_le_bytes(s));
                samples = Some(pcm.collect());
            }
            _ => {}
        }
        at += 8 + size + size % 2;
    }
    match (format, samples) {
        (Some((1, channels, rate, 16)), Some(samples)) => Ok(Pcm {
            rate,
            channels,
            samples,
        }),
        (Some(_), Some(_)) => Err(invalid("not 16-bit PCM")),
        _ => Err(invalid("no fmt or data chunk")),
    }
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
