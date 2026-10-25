use crate::{Config, Error};

const AUDIO_START: u32 = 101;
const REF_AUDIO_START: u32 = 103;
const REF_AUDIO_END: u32 = 104;

/// What the LMs are prompted with: positions that are each a text token or a patch of audio
/// latents.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    /// Each position's token, 0 where it is audio.
    pub tokens: Vec<u32>,
    /// Each position's patch, `patch_size` frames of `feat_dim` channels, zeros where it is a
    /// token.
    pub patches: Vec<f32>,
    /// Whether each position is audio, embedded from its patch rather than its token.
    pub audio: Vec<bool>,
}

impl Prompt {
    /// The prompt nanovllm-voxcpm builds to speak in a voice whose latents it is given both as
    /// reference audio and as audio to continue, as the Python service gave every voice: the
    /// voice's patches between the reference's start and end tokens, `text_tokens` (those of
    /// the voice's transcript and the text to speak, tokenized together), the audio start
    /// token, and the voice's patches again.
    ///
    /// `voice` is the voice's latent frames of `feat_dim` channels, one after another.
    ///
    /// # Errors
    ///
    /// If `voice` is not a whole number of patches.
    pub fn cloning(text_tokens: &[u32], voice: &[f32], config: &Config) -> Result<Self, Error> {
        let patch = config.patch_size * config.feat_dim;
        if !voice.len().is_multiple_of(patch) {
            return Err(Error::Request(format!(
                "a voice of {} values, not a whole number of {}-frame patches of {} channels",
                voice.len(),
                config.patch_size,
                config.feat_dim
            )));
        }
        let mut prompt = Self {
            tokens: vec![],
            patches: vec![],
            audio: vec![],
        };
        prompt.push_token(REF_AUDIO_START, patch);
        prompt.push_audio(voice, patch);
        prompt.push_token(REF_AUDIO_END, patch);
        for &token in text_tokens.iter().chain(&[AUDIO_START]) {
            prompt.push_token(token, patch);
        }
        prompt.push_audio(voice, patch);
        Ok(prompt)
    }

    /// The number of positions.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether there are no positions.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    fn push_token(&mut self, token: u32, patch: usize) {
        self.tokens.push(token);
        self.patches.resize(self.patches.len() + patch, 0.0);
        self.audio.push(false);
    }

    fn push_audio(&mut self, frames: &[f32], patch: usize) {
        let patches = frames.len() / patch;
        self.tokens.resize(self.tokens.len() + patches, 0);
        self.patches.extend_from_slice(frames);
        self.audio.resize(self.audio.len() + patches, true);
    }
}
