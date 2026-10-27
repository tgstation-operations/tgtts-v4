mod chunks;
mod common;

use audiovae::{AudioVaeConfig, DecodeSlots, DecodeStream, Decoder, Error};
use burn::tensor::{FloatDType, Int, Tensor};
use chunks::{Draws, streamed};
use common::{tiny, weights};
use testkit::{assert_close, device, fixture};

#[test]
fn decodes_as_upstream_in_the_bucket_of_out_sample_rate() {
    let device = device();
    let decoder = Decoder::load(&tiny(), &weights(), FloatDType::F32, &device).unwrap();
    let fixture = fixture!("audiovae");
    let audio = decoder.decode(fixture.tensor("input.decode.z", &device));
    // The output is within (-1, 1). Upstream's own f32 result is 2e-5 from its f64 one: the
    // random weights amplify rounding through thirteen convolutions.
    assert_close(audio, fixture.tensor("output.decode", &device), 1e-4, 0.0);
}

#[test]
fn streams_as_upstreams_streaming_decoder() {
    let device = device();
    let decoder = Decoder::load(&tiny(), &weights(), FloatDType::F32, &device).unwrap();
    let fixture = fixture!("audiovae");
    let latents = fixture.tensor("input.decode.z", &device);
    let audio = streamed(&decoder, latents, &[1, 1, 2, 4]);
    // Upstream's streamed audio is 4.8e-6 from its whole, and this 1.9e-5 from it: rounding
    // that the random weights amplify, as they do decoding whole.
    let expected = fixture.tensor("output.decode_streamed", &device);
    assert_close(audio, expected, 1e-4, 0.0);
}

#[test]
fn streams_in_chunks_of_any_length_as_it_decodes_whole() {
    let device = device();
    // In f64, where chunks of a frame come within 5e-13 of the whole: the random weights
    // amplify rounding thousands of times, to 4e-4 in f32.
    let decoder = Decoder::load(&tiny(), &weights(), FloatDType::F64, &device).unwrap();
    let mut draws = Draws::new(3);
    // Longer than the decoder reaches back: 54 frames before a sample's own.
    let frames = 64;
    let latents = draws.latents([2, 4, frames], &device);
    let whole = decoder.decode(latents.clone());
    let mut chunkings = vec![vec![1; frames], vec![frames]];
    chunkings.extend((0..4).map(|_| draws.lengths(frames, 9)));
    for lengths in chunkings {
        let audio = streamed(&decoder, latents.clone(), &lengths);
        assert_close(audio, whole.clone(), 1e-12, 0.0);
    }
}

/// Two streams decoded together a chunk at a time, the second starting a chunk after the first:
/// each comes out bit for bit as decoded alone in the same chunks.
#[test]
fn streams_decoded_together_come_out_as_alone() {
    let device = device();
    let decoder = Decoder::load(&tiny(), &weights(), FloatDType::F64, &device).unwrap();
    let mut draws = Draws::new(4);
    let (chunk, chunks) = (6, 4);
    let latents = draws.latents([2, 4, chunk * chunks], &device);
    let at = |row: usize, i: usize| {
        latents
            .clone()
            .narrow(0, row, 1)
            .narrow(2, i * chunk, chunk)
    };
    let alone = |row: usize, chunks: usize| {
        let mut stream = decoder.stream();
        let audio = (0..chunks).map(|i| stream.push(at(row, i)));
        Tensor::cat(audio.collect(), 2)
    };
    let (first, second) = (alone(0, chunks), alone(1, chunks - 1));
    let (mut one, mut two) = (decoder.stream(), decoder.stream());
    let mut audio = vec![one.push(at(0, 0))];
    let mut together = vec![];
    for i in 1..chunks {
        let both = Tensor::cat(vec![at(0, i), at(1, i - 1)], 0);
        let decoded = DecodeStream::push_together(&mut [&mut one, &mut two], both);
        audio.push(decoded.clone().narrow(0, 0, 1));
        together.push(decoded.narrow(0, 1, 1));
    }
    assert_close(Tensor::cat(audio, 2), first, 0.0, 0.0);
    assert_close(Tensor::cat(together, 2), second, 0.0, 0.0);
}

/// Streams in slots, decoded together a chunk at a time: the first alone, then beside the
/// second, which starts a chunk later in a slot another stream left, with rows of a spare slot
/// padding the batch. Each comes out bit for bit as decoded alone.
#[test]
fn streams_in_slots_come_out_as_alone() {
    let device = device();
    let decoder = Decoder::load(&tiny(), &weights(), FloatDType::F64, &device).unwrap();
    let mut draws = Draws::new(6);
    let (chunk, chunks) = (6, 4);
    let latents = draws.latents([2, 4, chunk * chunks], &device);
    let at = |row: usize, i: usize| {
        latents
            .clone()
            .narrow(0, row, 1)
            .narrow(2, i * chunk, chunk)
    };
    let alone = |row: usize, chunks: usize| {
        let mut stream = decoder.stream();
        let audio = (0..chunks).map(|i| stream.push(at(row, i)));
        Tensor::cat(audio.collect(), 2)
    };
    let (first, second) = (alone(0, chunks), alone(1, chunks - 1));
    let mut slots = DecodeSlots::new(&decoder, 3, &device);
    let slot = |slots: &[i64]| Tensor::<1, Int>::from_data(slots, &device);
    slots.push(slot(&[1]), at(1, 2));
    slots.clear(1);
    let mut audio = vec![slots.push(slot(&[0]), at(0, 0))];
    let mut together = vec![];
    for i in 1..chunks {
        let spare = latents.clone().narrow(0, 0, 1).narrow(2, 0, chunk);
        let rows = Tensor::cat(vec![at(0, i), at(1, i - 1), spare.clone(), spare], 0);
        let decoded = slots.push(slot(&[0, 1, 2, 2]), rows);
        audio.push(decoded.clone().narrow(0, 0, 1));
        together.push(decoded.narrow(0, 1, 1));
    }
    assert_close(Tensor::cat(audio, 2), first, 0.0, 0.0);
    assert_close(Tensor::cat(together, 2), second, 0.0, 0.0);
}

#[test]
fn boundaries_other_than_the_weights_are_refused() {
    let config = AudioVaeConfig {
        sr_bin_boundaries: Some(vec![16000, 24000, 32000]),
        ..tiny()
    };
    let error = Decoder::load(&config, &weights(), FloatDType::F32, &device()).unwrap_err();
    let Error::Weights(message) = error else {
        panic!("expected a weights error, got {error:?}");
    };
    assert_eq!(
        message,
        "decoder.sr_bin_boundaries: [20000, 30000, 40000], but the config has [16000, 24000, 32000]"
    );
}

#[test]
fn weights_of_another_shape_are_refused() {
    let config = AudioVaeConfig {
        decoder_dim: 64,
        ..tiny()
    };
    let error = Decoder::load(&config, &weights(), FloatDType::F32, &device()).unwrap_err();
    assert!(
        matches!(&error, Error::Weights(message) if message.contains("Shape mismatch")),
        "{error}"
    );
}
