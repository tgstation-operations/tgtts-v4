use std::ops::Range;

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

    /// The positions of the voice as reference audio, and the voice's patches there, if the
    /// prompt is as [`cloning`](Self::cloning) lays it out: the voice spoken again at the end,
    /// after the text and the audio start token.
    pub(crate) fn voice(&self) -> Option<(Range<usize>, &[f32])> {
        let len = self.len();
        let voice = self.audio.iter().skip(1).take_while(|&&a| a).count();
        let text = len.checked_sub(3 + 2 * voice)?;
        let patch = self.patches.len() / len.max(1);
        let reference = 1..1 + voice;
        let continuation = len - voice..len;
        let patches = |range: &Range<usize>| &self.patches[range.start * patch..range.end * patch];
        let laid_out = voice > 0
            && self.tokens[0] == REF_AUDIO_START
            && self.tokens[1 + voice] == REF_AUDIO_END
            && self.tokens[2 + voice + text] == AUDIO_START
            && self.audio[continuation.clone()].iter().all(|&a| a)
            && self.audio.iter().filter(|&&a| a).count() == 2 * voice
            && patches(&reference) == patches(&continuation);
        laid_out.then(|| (reference.clone(), patches(&reference)))
    }

    /// The positions of the text, between the reference audio and the audio start token, of a
    /// prompt [`voice`](Self::voice) finds laid out.
    pub(crate) fn text(&self) -> Range<usize> {
        let voice = self.audio.iter().skip(1).take_while(|&&a| a).count();
        2 + voice..self.len() - voice - 1
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
