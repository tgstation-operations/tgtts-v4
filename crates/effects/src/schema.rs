//! The effects a request asks for, as the game sends them in JSON: a list of graphs, run one
//! after another on mono audio at [`dsp::RATE`].
//!
//! A graph is either a chain, an array of nodes run in series, or an object of `nodes` by id and
//! `edges` between them, from the reserved `input` to the reserved `output`. A node is an object
//! naming its `effect` beside its parameters, whose names carry their units: `_hz`, `_db`, `_ms`
//! and `semitones`; a `ratio`, `q`, `depth`, `feedback`, `mix` or `weight` is a plain number.
//! Parameters are required unless a default is given, and a node with an unknown effect, an
//! unknown parameter or a value out of its range is refused, as is a graph that is cyclic or past
//! the limits in [`crate::graph`].

use std::{collections::BTreeMap, fmt};

use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, SeqAccess, Visitor, value::MapAccessDeserializer},
};

/// The id an edge leaves from to take a graph's input.
pub const INPUT: &str = "input";
/// The id an edge goes to to give a graph's output.
pub const OUTPUT: &str = "output";

/// A graph of effects.
#[derive(Clone, Debug, PartialEq)]
pub enum Graph {
    /// Nodes in series, the first taking the graph's input and the last giving its output; none
    /// leaves the audio as it is.
    Chain(Vec<Node>),
    /// Nodes wired by edges.
    ///
    /// An edge `[from, to]` feeds the output of `from`, a node or [`INPUT`], into `to`, a node or
    /// [`OUTPUT`]. Any node may feed several, but only a [`Node::Mix`] takes more than one: every
    /// other node, and the output, has exactly one edge in. There are no cycles, and every node
    /// is on a path from the input to the output. All branches start together, whatever each
    /// one's effects hold back, so a mix adds its inputs sample by sample from the start.
    Wired {
        /// The nodes by id: 1 to 32 characters of ASCII letters, digits, `_` and `-`, neither
        /// [`INPUT`] nor [`OUTPUT`].
        nodes: BTreeMap<String, Node>,
        /// The edges, each `[from, to]`.
        edges: Vec<[String; 2]>,
    },
}

/// An effect and its parameters, `{"effect": NAME, ...}`, NAME as each is listed here in lower
/// case.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "effect", rename_all = "lowercase", deny_unknown_fields)]
pub enum Node {
    /// A gain of `gain_db`, in [-60, 40].
    Gain {
        /// The gain, in dB.
        gain_db: f64,
    },
    /// A second-order lowpass, falling 12 dB an octave above `frequency_hz`, in [20, 20000], with
    /// the peak `q` gives it there, in [0.1, 20], 0.7071 by default, which has none.
    Lowpass {
        /// The cutoff, in Hz.
        frequency_hz: f64,
        /// The RBJ cookbook's Q.
        #[serde(default = "butterworth")]
        q: f64,
    },
    /// A second-order highpass, the mirror of [`Node::Lowpass`].
    Highpass {
        /// The cutoff, in Hz.
        frequency_hz: f64,
        /// The RBJ cookbook's Q.
        #[serde(default = "butterworth")]
        q: f64,
    },
    /// A peaking equalizer: `gain_db`, in [-60, 60], at `frequency_hz`, over a width `q` gives,
    /// in [0.1, 20], 0.7071 by default.
    Peak {
        /// The center, in Hz.
        frequency_hz: f64,
        /// The gain at the center, in dB.
        gain_db: f64,
        /// The RBJ cookbook's Q.
        #[serde(default = "butterworth")]
        q: f64,
    },
    /// A low shelf: `gain_db`, in [-60, 60], below `frequency_hz`, the midpoint of its slope,
    /// which `q` sets the steepness of, in [0.1, 20], 0.7071 by default.
    Lowshelf {
        /// The midpoint, in Hz.
        frequency_hz: f64,
        /// The gain of the shelf, in dB.
        gain_db: f64,
        /// The RBJ cookbook's Q.
        #[serde(default = "butterworth")]
        q: f64,
    },
    /// A high shelf, the mirror of [`Node::Lowshelf`].
    Highshelf {
        /// The midpoint, in Hz.
        frequency_hz: f64,
        /// The gain of the shelf, in dB.
        gain_db: f64,
        /// The RBJ cookbook's Q.
        #[serde(default = "butterworth")]
        q: f64,
    },
    /// A compressor on the RMS level: above `threshold_db`, in [-80, 0], the level rises by
    /// 1 / `ratio`, in [1, 100], dB a dB, the bend spread over `knee_db`, in [0, 24], 0 by
    /// default; its gain follows a rising level within `attack_ms`, in [0.1, 1000], and a
    /// falling one within `release_ms`, in [1, 5000], and `makeup_db`, in [0, 40], is added.
    Compressor {
        /// The level compression starts at, in dBFS.
        threshold_db: f64,
        /// The ratio of the rise in level to the rise in output above the threshold.
        ratio: f64,
        /// The time constant of the gain coming down, in ms.
        attack_ms: f64,
        /// The time constant of the gain going back up, in ms.
        release_ms: f64,
        /// The gain after it, in dB.
        makeup_db: f64,
        /// The width of the bend at the threshold, in dB.
        #[serde(default)]
        knee_db: f64,
    },
    /// A limiter holding the true peak at or under `ceiling_db`, in [-40, 0], with 5 ms of
    /// lookahead and an 80 ms release.
    Limiter {
        /// The ceiling, in dBTP.
        ceiling_db: f64,
    },
    /// Soft clipping, `tanh(drive * x)`, `drive_db` in [0, 40].
    Saturate {
        /// The drive, in dB.
        drive_db: f64,
    },
    /// A sample-rate crusher: the samples held at `rate_hz`, in [100, 48000], with nothing to
    /// stop what that folds over, then mixed with the signal as it was, `mix`, in [0, 1], being
    /// the crushed part.
    Crush {
        /// The rate the crushed samples change at, in Hz.
        rate_hz: f64,
        /// The fraction of the output that is crushed.
        mix: f64,
    },
    /// The pitch shifted by `semitones`, in [-24, 24], the duration kept. The formants move with
    /// the pitch, as a voice played faster would, unless `keep_formants`, false by default, keeps
    /// them where they were, as the same voice speaking higher would.
    Pitch {
        /// The shift, in semitones, up where positive.
        semitones: f64,
        /// Whether the formants stay where they are.
        #[serde(default)]
        keep_formants: bool,
    },
    /// The tempo times `ratio`, in [0.25, 4], the pitch kept: the duration divided by it.
    Tempo {
        /// The ratio of the new tempo to the old.
        ratio: f64,
    },
    /// Tape speed times `ratio`, in [0.25, 4]: pitch and tempo together, the duration divided by
    /// it.
    Speed {
        /// The ratio of the new speed to the old.
        ratio: f64,
    },
    /// The gain swinging sinusoidally between 1 - `depth`, in [0, 1], and 1, `rate_hz` times a
    /// second, in [0.1, 50].
    Tremolo {
        /// How often the gain swings, in Hz.
        rate_hz: f64,
        /// How far down it swings.
        depth: f64,
    },
    /// The pitch swinging sinusoidally by up to `depth_semitones` either way, in [0, 3],
    /// `rate_hz` times a second, in [0.1, 20].
    Vibrato {
        /// How often the pitch swings, in Hz.
        rate_hz: f64,
        /// How far it swings each way, in semitones.
        depth_semitones: f64,
    },
    /// A swept feedback comb, `y = x + feedback * y` delayed, `feedback` in [-0.95, 0.95], the
    /// delay sweeping between none and `delay_ms`, in [0.1, 20], `rate_hz` times a second, in
    /// [0.01, 10], along `waveform`.
    Flanger {
        /// The longest delay, in ms.
        delay_ms: f64,
        /// How often the delay sweeps, in Hz.
        rate_hz: f64,
        /// The part of the delayed output fed back.
        feedback: f64,
        /// The shape of the sweep.
        #[serde(default)]
        waveform: Waveform,
    },
    /// A resonant lowpass whose cutoff swings sinusoidally between `center_hz` - `swing_hz` and
    /// `center_hz` + `swing_hz`, `rate_hz` times a second: the center in [20, 20000], the swing
    /// in [0, center - 20], the rate in [0.1, 50], and the resonance `q` in [0.5, 20], 0.7071 by
    /// default, which has none.
    Wobble {
        /// The cutoff's middle, in Hz.
        center_hz: f64,
        /// How far the cutoff swings each way, in Hz.
        swing_hz: f64,
        /// How often it swings, in Hz.
        rate_hz: f64,
        /// The resonance at the cutoff, as a filter's Q.
        #[serde(default = "butterworth")]
        q: f64,
    },
    /// The sum of delayed copies, 1 to 16 `taps`; one at 0 ms is the signal itself. The audio
    /// grows by the longest delay.
    Echo {
        /// The copies.
        taps: Vec<Tap>,
    },
    /// The signal convolved with one of the service's impulse responses, scaled to unit energy,
    /// so that white noise comes out as loud as it went in. Only the convolution is given, and
    /// the audio grows by the impulse's length.
    Convolve {
        /// Which impulse.
        impulse: Impulse,
    },
    /// White noise added at `level_db` RMS, in [-120, -20], as long as the signal.
    Noise {
        /// The noise's RMS level, in dBFS.
        level_db: f64,
    },
    /// The sum of a [`Graph::Wired`]'s nodes, by `weights` keyed by the id of each node, or
    /// [`INPUT`], with an edge into it; 1 to 16 of them, each in [-4, 4]. It lasts as long as its
    /// longest input.
    Mix {
        /// The weight of each input, by the id it comes from.
        weights: BTreeMap<String, f64>,
    },
}

/// A copy an [`Node::Echo`] adds.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Tap {
    /// Its delay, in ms, in [0, 2000].
    pub delay_ms: f64,
    /// Its gain, in [-4, 4].
    pub weight: f64,
}

/// The shape a [`Node::Flanger`]'s delay sweeps along.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Waveform {
    /// A sine.
    #[default]
    Sine,
    /// A triangle: the delay changes at a steady rate.
    Triangle,
}

/// The impulse responses a [`Node::Convolve`] takes.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Impulse {
    /// SynthImpulse.wav, a fifth of a second of synthetic resonance.
    Synth,
}

fn butterworth() -> f64 {
    std::f64::consts::FRAC_1_SQRT_2
}

impl Node {
    /// Whether its parameters are in their ranges, and if not, which is not.
    pub fn check(&self) -> Result<(), String> {
        match *self {
            Self::Gain { gain_db } => within("gain_db", gain_db, -60.0, 40.0),
            Self::Lowpass { frequency_hz, q } | Self::Highpass { frequency_hz, q } => {
                frequency(frequency_hz)?;
                within("q", q, 0.1, 20.0)
            }
            Self::Peak {
                frequency_hz,
                gain_db,
                q,
            }
            | Self::Lowshelf {
                frequency_hz,
                gain_db,
                q,
            }
            | Self::Highshelf {
                frequency_hz,
                gain_db,
                q,
            } => {
                frequency(frequency_hz)?;
                within("gain_db", gain_db, -60.0, 60.0)?;
                within("q", q, 0.1, 20.0)
            }
            Self::Compressor {
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                makeup_db,
                knee_db,
            } => {
                within("threshold_db", threshold_db, -80.0, 0.0)?;
                within("ratio", ratio, 1.0, 100.0)?;
                within("attack_ms", attack_ms, 0.1, 1000.0)?;
                within("release_ms", release_ms, 1.0, 5000.0)?;
                within("makeup_db", makeup_db, 0.0, 40.0)?;
                within("knee_db", knee_db, 0.0, 24.0)
            }
            Self::Limiter { ceiling_db } => within("ceiling_db", ceiling_db, -40.0, 0.0),
            Self::Saturate { drive_db } => within("drive_db", drive_db, 0.0, 40.0),
            Self::Crush { rate_hz, mix } => {
                within("rate_hz", rate_hz, 100.0, 48000.0)?;
                within("mix", mix, 0.0, 1.0)
            }
            Self::Pitch { semitones, .. } => within("semitones", semitones, -24.0, 24.0),
            Self::Tempo { ratio } | Self::Speed { ratio } => within("ratio", ratio, 0.25, 4.0),
            Self::Tremolo { rate_hz, depth } => {
                within("rate_hz", rate_hz, 0.1, 50.0)?;
                within("depth", depth, 0.0, 1.0)
            }
            Self::Vibrato {
                rate_hz,
                depth_semitones,
            } => {
                within("rate_hz", rate_hz, 0.1, 20.0)?;
                within("depth_semitones", depth_semitones, 0.0, 3.0)
            }
            Self::Flanger {
                delay_ms,
                rate_hz,
                feedback,
                ..
            } => {
                within("delay_ms", delay_ms, 0.1, 20.0)?;
                within("rate_hz", rate_hz, 0.01, 10.0)?;
                within("feedback", feedback, -0.95, 0.95)
            }
            Self::Wobble {
                center_hz,
                swing_hz,
                rate_hz,
                q,
            } => {
                within("center_hz", center_hz, 20.0, 20000.0)?;
                within("swing_hz", swing_hz, 0.0, center_hz - 20.0)?;
                within("rate_hz", rate_hz, 0.1, 50.0)?;
                within("q", q, 0.5, 20.0)
            }
            Self::Echo { ref taps } => {
                if !(1..=16).contains(&taps.len()) {
                    return Err(format!("{} taps, where 1 to 16 are allowed", taps.len()));
                }
                taps.iter().try_for_each(|tap| {
                    within("delay_ms", tap.delay_ms, 0.0, 2000.0)?;
                    within("weight", tap.weight, -4.0, 4.0)
                })
            }
            Self::Convolve { .. } => Ok(()),
            Self::Noise { level_db } => within("level_db", level_db, -120.0, -20.0),
            Self::Mix { ref weights } => {
                if !(1..=16).contains(&weights.len()) {
                    return Err(format!(
                        "{} weights, where 1 to 16 are allowed",
                        weights.len()
                    ));
                }
                weights
                    .values()
                    .try_for_each(|&weight| within("weight", weight, -4.0, 4.0))
            }
        }
    }
}

fn frequency(frequency_hz: f64) -> Result<(), String> {
    within("frequency_hz", frequency_hz, 20.0, 20000.0)
}

fn within(name: &str, value: f64, low: f64, high: f64) -> Result<(), String> {
    if (low..=high).contains(&value) {
        Ok(())
    } else {
        Err(format!("{name} {value} is not in [{low}, {high}]"))
    }
}

impl<'de> Deserialize<'de> for Graph {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(GraphVisitor)
    }
}

struct GraphVisitor;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wired {
    nodes: BTreeMap<String, Node>,
    edges: Vec<[String; 2]>,
}

impl<'de> Visitor<'de> for GraphVisitor {
    type Value = Graph;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an array of nodes, or an object of nodes and edges")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Graph, A::Error> {
        let mut nodes = Vec::new();
        while let Some(node) = seq.next_element()? {
            nodes.push(node);
        }
        Ok(Graph::Chain(nodes))
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Graph, A::Error> {
        let Wired { nodes, edges } = Wired::deserialize(MapAccessDeserializer::new(map))?;
        Ok(Graph::Wired { nodes, edges })
    }
}
