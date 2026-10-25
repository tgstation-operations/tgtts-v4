//! Graphs of effects checked and run: a [`Pipeline`] of the graphs a request gives, one after
//! another.
//!
//! A request is refused past [`GRAPHS`] graphs, [`NODES`] nodes in a graph or [`ALL_NODES`] in
//! all, or [`EDGES`] edges in a graph, or where its effects would make the audio more than
//! [`LONGER`] times as long and [`LONGER_BY`] seconds besides.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error, fmt,
};

use crate::{
    Processor, RATE,
    convolve::{Convolve, Impulses},
    delay::{Echo, Flanger, Vibrato},
    dynamics::{Compressor, Limiter},
    filter::{Biquad, Shape, Wobble},
    level::{Gain, Noise, Tremolo},
    schema::{Graph, INPUT, Node, OUTPUT},
    shape::{Crush, Saturate},
    time::{Pitch, Speed, Tempo},
};

/// The most graphs a request can give.
pub const GRAPHS: usize = 16;
/// The most nodes a graph can have.
pub const NODES: usize = 32;
/// The most nodes a request can give in all.
pub const ALL_NODES: usize = 64;
/// The most edges a graph can have.
pub const EDGES: usize = 128;
/// The most times longer the effects can make the audio, besides [`LONGER_BY`].
pub const LONGER: f64 = 4.0;
/// The most seconds the effects can add to the audio, besides [`LONGER`].
pub const LONGER_BY: f64 = 5.0;

/// Why a request's effects were refused: where, and what is wrong there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// The index of the graph, where one is to blame.
    pub graph: Option<usize>,
    /// The node: its id in a wired graph, its index in a chain.
    pub node: Option<String>,
    /// What is wrong.
    pub reason: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.graph {
            Some(graph) => write!(f, "effects[{graph}]")?,
            None => write!(f, "effects")?,
        }
        if let Some(node) = &self.node {
            write!(f, ", node {node}")?;
        }
        write!(f, ": {}", self.reason)
    }
}

impl error::Error for Error {}

/// The graphs of a request, checked and ready to run one after another, as one [`Processor`].
pub struct Pipeline {
    stages: Vec<Stage>,
}

impl Pipeline {
    /// `effects` checked and built, with the service's `impulses` for their convolutions.
    pub fn new(effects: &[Graph], impulses: &Impulses) -> Result<Self, Error> {
        let refuse = |reason: String| Error {
            graph: None,
            node: None,
            reason,
        };
        if effects.len() > GRAPHS {
            return Err(refuse(format!(
                "{} graphs, where at most {GRAPHS} are allowed",
                effects.len()
            )));
        }
        let plans = effects
            .iter()
            .enumerate()
            .map(|(index, graph)| {
                Plan::new(graph, impulses).map_err(|(node, reason)| Error {
                    graph: Some(index),
                    node,
                    reason,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let nodes: usize = plans.iter().map(|plan| plan.steps.len()).sum();
        if nodes > ALL_NODES {
            return Err(refuse(format!(
                "{nodes} nodes, where at most {ALL_NODES} are allowed in all"
            )));
        }
        let (times, by) = plans.iter().fold((1.0, 0.0), |(times, by), plan| {
            (
                times * plan.length.times,
                by * plan.length.times + plan.length.by,
            )
        });
        if times > LONGER + 1e-9 || by > LONGER_BY + 1e-9 {
            return Err(refuse(format!(
                "the audio would be {times:.2} times as long and {by:.2} s longer, where \
                 {LONGER} times and {LONGER_BY} s are allowed"
            )));
        }
        let stages = plans.into_iter().map(|plan| plan.build(impulses)).collect();
        Ok(Self { stages })
    }
}

impl Processor for Pipeline {
    fn latency(&self) -> usize {
        self.stages.iter().map(Stage::latency).sum()
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let mut chunk = input.to_vec();
        for stage in &mut self.stages {
            chunk = stage.run(chunk, false);
        }
        output.extend(chunk);
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let mut chunk = Vec::new();
        for stage in &mut self.stages {
            chunk = stage.run(chunk, true);
        }
        output.extend(chunk);
    }
}

/// How much longer a stretch of a graph makes the audio: `times` as long, then `by` seconds.
#[derive(Clone, Copy, Debug)]
struct Length {
    times: f64,
    by: f64,
}

/// A graph checked: its nodes in an order where each comes after what feeds it.
struct Plan<'a> {
    steps: Vec<(&'a Node, Vec<usize>)>,
    /// The value the output takes: 0 for the input, `i + 1` for step `i`.
    output: usize,
    length: Length,
}

type Refusal = (Option<String>, String);

impl<'a> Plan<'a> {
    fn new(graph: &'a Graph, impulses: &Impulses) -> Result<Self, Refusal> {
        let (ids, nodes, edges): (Vec<String>, Vec<&Node>, Vec<[String; 2]>) = match graph {
            Graph::Chain(chain) => {
                let ids: Vec<String> = (0..chain.len()).map(|i| i.to_string()).collect();
                let mut path = vec![INPUT.to_owned()];
                path.extend(ids.iter().cloned());
                path.push(OUTPUT.to_owned());
                let edges = path.windows(2).map(|w| [w[0].clone(), w[1].clone()]);
                (ids, chain.iter().collect(), edges.collect())
            }
            Graph::Wired { nodes, edges } => {
                for id in nodes.keys() {
                    let valid = (1..=32).contains(&id.len())
                        && id
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                        && id != INPUT
                        && id != OUTPUT;
                    if !valid {
                        return Err((
                            Some(format!("{id:?}")),
                            "is not 1 to 32 letters, digits, _ or -, other than input and \
                             output"
                                .to_owned(),
                        ));
                    }
                }
                if edges.len() > EDGES {
                    return Err((
                        None,
                        format!("{} edges, where at most {EDGES} are allowed", edges.len()),
                    ));
                }
                (
                    nodes.keys().cloned().collect(),
                    nodes.values().collect(),
                    edges.clone(),
                )
            }
        };
        let named = |i: usize| match graph {
            Graph::Chain(_) => Some(ids[i].clone()),
            Graph::Wired { .. } => Some(format!("{:?}", ids[i])),
        };
        if nodes.len() > NODES {
            return Err((
                None,
                format!("{} nodes, where at most {NODES} are allowed", nodes.len()),
            ));
        }
        for (i, node) in nodes.iter().enumerate() {
            node.check().map_err(|reason| (named(i), reason))?;
        }

        // Values: 0 the input, i + 1 node i; the output is a sink of its own.
        let index: BTreeMap<&str, usize> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i + 1))
            .collect();
        let sink = nodes.len() + 1;
        let mut sources: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nodes.len() + 2];
        for [from, to] in &edges {
            let edge = || Some(format!("edge [{from:?}, {to:?}]"));
            let from_value = match from.as_str() {
                INPUT => 0,
                id => *index
                    .get(id)
                    .ok_or_else(|| (edge(), format!("there is no node {from:?}")))?,
            };
            let to_value = match to.as_str() {
                OUTPUT => sink,
                id => *index
                    .get(id)
                    .ok_or_else(|| (edge(), format!("there is no node {to:?}")))?,
            };
            if !sources[to_value].insert(from_value) {
                return Err((edge(), "is given twice".to_owned()));
            }
        }
        if sources[sink].len() != 1 {
            return Err((
                None,
                format!(
                    "the output has {} edges in, where it takes exactly one",
                    sources[sink].len()
                ),
            ));
        }
        let name = |value: usize| if value == 0 { INPUT } else { &ids[value - 1] };
        for (i, node) in nodes.iter().enumerate() {
            let fed = &sources[i + 1];
            if let Node::Mix { weights } = node {
                let fed: BTreeSet<&str> = fed.iter().map(|&v| name(v)).collect();
                let weighted: BTreeSet<&str> = weights.keys().map(String::as_str).collect();
                if fed != weighted {
                    return Err((
                        named(i),
                        format!("is weighted for {weighted:?}, but has edges in from {fed:?}"),
                    ));
                }
            } else if fed.is_empty() {
                return Err((named(i), "has no edge in".to_owned()));
            } else if fed.len() > 1 {
                return Err((
                    named(i),
                    format!(
                        "has {} edges in, where only a mix takes more than one",
                        fed.len()
                    ),
                ));
            }
        }

        let mut feeds: Vec<Vec<usize>> = vec![Vec::new(); nodes.len() + 2];
        for (to, from) in sources.iter().enumerate() {
            for &from in from {
                feeds[from].push(to);
            }
        }
        let reached = |start: usize, next: &dyn Fn(usize) -> Vec<usize>| {
            let mut seen = vec![false; nodes.len() + 2];
            let mut queue = VecDeque::from([start]);
            seen[start] = true;
            while let Some(v) = queue.pop_front() {
                for w in next(v) {
                    if !seen[w] {
                        seen[w] = true;
                        queue.push_back(w);
                    }
                }
            }
            seen
        };
        let from_input = reached(0, &|v| feeds[v].clone());
        let to_output = reached(sink, &|v| sources[v].iter().copied().collect());
        for i in 0..nodes.len() {
            if !from_input[i + 1] {
                return Err((named(i), "is not fed from the input".to_owned()));
            }
            if !to_output[i + 1] {
                return Err((named(i), "does not lead to the output".to_owned()));
            }
        }

        // Kahn's algorithm: what is left once nothing more is ready is on a cycle.
        let mut waiting: Vec<usize> = sources.iter().map(BTreeSet::len).collect();
        let mut ready = VecDeque::from([0]);
        let mut order = Vec::with_capacity(nodes.len());
        while let Some(v) = ready.pop_front() {
            if (1..sink).contains(&v) {
                order.push(v - 1);
            }
            for &w in &feeds[v] {
                waiting[w] -= 1;
                if waiting[w] == 0 {
                    ready.push_back(w);
                }
            }
        }
        if let Some(i) = (0..nodes.len()).find(|&i| waiting[i + 1] > 0) {
            return Err((named(i), "is on a cycle".to_owned()));
        }

        let mut position = vec![0; nodes.len() + 1];
        for (step, &i) in order.iter().enumerate() {
            position[i + 1] = step + 1;
        }
        let mut lengths = vec![
            Length {
                times: 1.0,
                by: 0.0
            };
            order.len() + 1
        ];
        let steps: Vec<(&Node, Vec<usize>)> = order
            .iter()
            .enumerate()
            .map(|(step, &i)| {
                let node = nodes[i];
                let from: Vec<usize> = match node {
                    // In the order of the weights, which the mix is built from.
                    Node::Mix { weights } => weights
                        .keys()
                        .map(|id| match id.as_str() {
                            INPUT => 0,
                            id => position[index[id]],
                        })
                        .collect(),
                    _ => sources[i + 1].iter().map(|&v| position[v]).collect(),
                };
                let fed = from.iter().map(|&v| lengths[v]).fold(
                    Length {
                        times: 0.0,
                        by: 0.0,
                    },
                    |a, b| Length {
                        times: a.times.max(b.times),
                        by: a.by.max(b.by),
                    },
                );
                lengths[step + 1] = lengthen(node, fed, impulses);
                (node, from)
            })
            .collect();
        let output = position[*sources[sink].first().expect("checked above")];
        Ok(Self {
            steps,
            output,
            length: lengths[output],
        })
    }

    fn build(self, impulses: &Impulses) -> Stage {
        let steps = self
            .steps
            .into_iter()
            .map(|(node, from)| match node {
                Node::Mix { weights } => Step::Mix {
                    sources: from
                        .iter()
                        .zip(weights.values())
                        .map(|(&v, &w)| (v, w as f32))
                        .collect(),
                    queues: vec![VecDeque::new(); from.len()],
                },
                node => Step::Effect {
                    source: from[0],
                    processor: processor(node, impulses),
                },
            })
            .collect();
        Stage {
            steps,
            output: self.output,
        }
    }
}

/// How much longer `node` makes audio that `fed` says how much longer it already is.
fn lengthen(node: &Node, fed: Length, impulses: &Impulses) -> Length {
    match *node {
        Node::Tempo { ratio } | Node::Speed { ratio } => Length {
            times: fed.times / ratio,
            by: fed.by / ratio,
        },
        Node::Echo { ref taps } => Length {
            times: fed.times,
            by: fed.by + taps.iter().map(|t| t.delay_ms).fold(0.0, f64::max) / 1000.0,
        },
        Node::Convolve { impulse } => Length {
            times: fed.times,
            by: fed.by + impulses.get(impulse).len() as f64 / RATE,
        },
        _ => fed,
    }
}

/// The processor for `node`, which is not a mix and has been checked.
fn processor(node: &Node, impulses: &Impulses) -> Box<dyn Processor> {
    match *node {
        Node::Gain { gain_db } => Box::new(Gain::new(gain_db)),
        Node::Lowpass { frequency_hz, q } => Box::new(Biquad::new(Shape::Lowpass, frequency_hz, q)),
        Node::Highpass { frequency_hz, q } => {
            Box::new(Biquad::new(Shape::Highpass, frequency_hz, q))
        }
        Node::Peak {
            frequency_hz,
            gain_db,
            q,
        } => Box::new(Biquad::new(Shape::Peak(gain_db), frequency_hz, q)),
        Node::Lowshelf {
            frequency_hz,
            gain_db,
            q,
        } => Box::new(Biquad::new(Shape::Lowshelf(gain_db), frequency_hz, q)),
        Node::Highshelf {
            frequency_hz,
            gain_db,
            q,
        } => Box::new(Biquad::new(Shape::Highshelf(gain_db), frequency_hz, q)),
        Node::Compressor {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            knee_db,
        } => Box::new(Compressor::new(
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            knee_db,
        )),
        Node::Limiter { ceiling_db } => Box::new(Limiter::new(ceiling_db)),
        Node::Saturate { drive_db } => Box::new(Saturate::new(drive_db)),
        Node::Crush { rate_hz, mix } => Box::new(Crush::new(rate_hz, mix)),
        Node::Pitch {
            semitones,
            keep_formants,
        } => Box::new(Pitch::new(semitones, keep_formants)),
        Node::Tempo { ratio } => Box::new(Tempo::new(ratio)),
        Node::Speed { ratio } => Box::new(Speed::new(ratio)),
        Node::Tremolo { rate_hz, depth } => Box::new(Tremolo::new(rate_hz, depth)),
        Node::Vibrato {
            rate_hz,
            depth_semitones,
        } => Box::new(Vibrato::new(rate_hz, depth_semitones)),
        Node::Flanger {
            delay_ms,
            rate_hz,
            feedback,
            waveform,
        } => Box::new(Flanger::new(delay_ms, rate_hz, feedback, waveform)),
        Node::Wobble {
            center_hz,
            swing_hz,
            rate_hz,
            q,
        } => Box::new(Wobble::new(center_hz, swing_hz, rate_hz, q)),
        Node::Echo { ref taps } => {
            Box::new(Echo::new(taps.iter().map(|tap| (tap.delay_ms, tap.weight))))
        }
        Node::Convolve { impulse } => Box::new(Convolve::new(impulses.get(impulse).clone())),
        Node::Noise { level_db } => Box::new(Noise::new(level_db)),
        Node::Mix { .. } => unreachable!("a mix is a step of its own"),
    }
}

/// A graph ready to run.
struct Stage {
    steps: Vec<Step>,
    output: usize,
}

enum Step {
    Effect {
        source: usize,
        processor: Box<dyn Processor>,
    },
    /// A mix of the values `sources` gives, each with its weight, holding what each has given
    /// until the others have given as much.
    Mix {
        sources: Vec<(usize, f32)>,
        queues: Vec<VecDeque<f32>>,
    },
}

impl Stage {
    fn latency(&self) -> usize {
        let mut latencies = vec![0; self.steps.len() + 1];
        for (i, step) in self.steps.iter().enumerate() {
            latencies[i + 1] = match step {
                Step::Effect { source, processor } => latencies[*source] + processor.latency(),
                Step::Mix { sources, .. } => sources
                    .iter()
                    .map(|&(v, _)| latencies[v])
                    .max()
                    .unwrap_or(0),
            };
        }
        latencies[self.output]
    }

    /// The output for `input`, and if `last`, everything still held.
    fn run(&mut self, input: Vec<f32>, last: bool) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.steps.len() + 1);
        values.push(input);
        for step in &mut self.steps {
            let mut out = Vec::new();
            match step {
                Step::Effect { source, processor } => {
                    processor.process(&values[*source], &mut out);
                    if last {
                        processor.finish(&mut out);
                    }
                }
                Step::Mix { sources, queues } => {
                    for (&(v, _), queue) in sources.iter().zip(queues.iter_mut()) {
                        queue.extend(&values[v]);
                    }
                    let lengths = queues.iter().map(VecDeque::len);
                    let ready = if last { lengths.max() } else { lengths.min() };
                    out = vec![0.0; ready.unwrap_or(0)];
                    for (&(_, weight), queue) in sources.iter().zip(queues.iter_mut()) {
                        let take = queue.len().min(out.len());
                        for (y, x) in out.iter_mut().zip(queue.drain(..take)) {
                            *y += weight * x;
                        }
                    }
                }
            }
            values.push(out);
        }
        std::mem::take(&mut values[self.output])
    }
}
