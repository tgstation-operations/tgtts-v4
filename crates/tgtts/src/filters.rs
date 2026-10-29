//! The ffmpeg filter graphs the game sends, run as the effects that take their place.
//!
//! The game joins the filters of a message with commas: its speaker's own, those its status
//! effects and a megaphone add, then its mask's, each a graph that can have commas of its own. A
//! graph made only of filters known here runs as their effects, in its order, each the effects
//! fixtures/effects.py compared with what ffmpeg makes of it; any other still goes to ffmpeg.

use std::{collections::HashSet, sync::Mutex};

use effects::{
    Pipeline,
    convolve::Impulses,
    schema::{self, Graph},
};
use serde_json::{Value, json};
use tracing::warn;

/// A filter the game sends, and the effects in its place.
struct Known {
    /// The filter as the game's code writes it. `%PITCH%`, `%FEMALE%` and `%BLIPS%` stand for
    /// the numbers the game writes in their place; `%SAMPLE_RATE%` the service fills.
    filter: &'static str,
    effects: fn(Filled) -> Value,
}

/// The numbers the game writes in a filter's placeholders: the speaker's pitch, 0 where pitch is
/// disabled, 1 for a voice whose name has "Woman" in it, and 1 for blips.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Filled {
    pitch: f64,
    female: bool,
    blips: bool,
}

const PLACEHOLDERS: [&str; 3] = ["%PITCH%", "%FEMALE%", "%BLIPS%"];

const SYNTHETIC: Known = Known {
    filter: "alimiter=0.9,acompressor=threshold=0.2:ratio=20:attack=10:release=50:makeup=2,\
             highpass=f=1000",
    effects: |_| json!([synthetic()]),
};

const TRAM: Known = Known {
    filter: "highpass=f=300,lowpass=f=3500,aecho=0.8:0.9:70|140:0.3|0.15,alimiter=0.9,\
             acompressor=threshold=0.2:ratio=20:attack=10:release=50:makeup=2,highpass=f=1000",
    effects: |_| {
        json!([
            [
                {"effect": "highpass", "frequency_hz": 300},
                {"effect": "lowpass", "frequency_hz": 3500},
                {"effect": "echo", "taps": [
                    {"delay_ms": 0, "weight": 0.72},
                    {"delay_ms": 70, "weight": 0.27},
                    {"delay_ms": 140, "weight": 0.135},
                ]},
            ],
            synthetic(),
        ])
    },
};

// ffmpeg refuses its ratio of 40, and its speech was served unfiltered.
const FISHING_CAPSULE: Known = Known {
    filter: "alimiter=0.9,acompressor=threshold=0.3:ratio=40:attack=15:release=350:makeup=1.5,\
             highpass=f=1000,rubberband=pitch=1.5",
    effects: |_| {
        json!([[
            {"effect": "gain", "gain_db": -0.915},
            {"effect": "limiter", "ceiling_db": 0},
            {"effect": "compressor", "threshold_db": -10.458, "ratio": 40, "attack_ms": 15,
                "release_ms": 350, "makeup_db": 3.522, "knee_db": 9.031},
            {"effect": "highpass", "frequency_hz": 1000},
            {"effect": "pitch", "semitones": 7.02},
        ]])
    },
};

const GAS_MASK: Known = Known {
    filter: "lowpass=f=750,volume=2",
    effects: |_| {
        json!([[
            {"effect": "lowpass", "frequency_hz": 750},
            {"effect": "gain", "gain_db": 6.021},
        ]])
    },
};

const MEGAPHONE: Known = Known {
    filter: "acrusher=samples=2:level_out=6,aecho=delays=90:decays=0.3,aemphasis=type=cd,\
             acontrast=30,crystalizer=i=5",
    effects: |_| {
        json!([[
            {"effect": "crush", "rate_hz": 24000, "mix": 0.5},
            {"effect": "gain", "gain_db": 16.0},
            {"effect": "echo", "taps": [
                {"delay_ms": 0, "weight": 0.18},
                {"delay_ms": 90, "weight": 0.09},
            ]},
            {"effect": "highshelf", "frequency_hz": 5784, "gain_db": -10.93, "q": 0.456},
            {"effect": "saturate", "amount": 0.3},
            {"effect": "highshelf", "frequency_hz": 4474, "gain_db": 20.83, "q": 0.422},
            {"effect": "clip", "ceiling_db": 0},
        ]])
    },
};

const MEGAPHONE_EMAGGED: Known = Known {
    filter: "acrusher=samples=9:level_out=7,aecho=delays=100:decays=0.4,aemphasis=type=emi,\
             crystalizer=i=6,acontrast=60,rubberband=pitch=0.9",
    effects: |_| {
        json!([[
            {"effect": "crush", "rate_hz": 5333.3, "mix": 0.5},
            {"effect": "gain", "gain_db": 16.902},
            {"effect": "echo", "taps": [
                {"delay_ms": 0, "weight": 0.18},
                {"delay_ms": 100, "weight": 0.12},
            ]},
            {"effect": "lowshelf", "frequency_hz": 202.2, "gain_db": 15.74, "q": 0.487},
            {"effect": "highshelf", "frequency_hz": 20000, "gain_db": -56.13, "q": 0.205},
            {"effect": "highshelf", "frequency_hz": 4134, "gain_db": 22.28, "q": 0.412},
            {"effect": "clip", "ceiling_db": 0},
            {"effect": "saturate", "amount": 0.6},
            {"effect": "pitch", "semitones": -1.824},
        ]])
    },
};

const SECHAILER: Known = Known {
    filter: "[0:a] asetrate=%SAMPLE_RATE%*0.7,aresample=16000,atempo=1/0.7,\
             lowshelf=g=-20:f=500,highpass=f=500,\
             aphaser=in_gain=1:out_gain=1:delay=3.0:decay=0.4:speed=0.5:type=t [out]; \
             [out]atempo=1.2,volume=15dB [final]; anoisesrc=a=0.01:d=60 [noise]; \
             [final][noise] amix=duration=shortest",
    effects: |_| {
        json!([[
            {"effect": "pitch", "semitones": -6.175, "method": "splice"},
            {"effect": "lowshelf", "frequency_hz": 500, "gain_db": -20, "q": 0.5},
            {"effect": "highpass", "frequency_hz": 500},
            {"effect": "flanger", "delay_ms": 3, "rate_hz": 0.5, "feedback": 0.4,
                "waveform": "triangle"},
            {"effect": "tempo", "ratio": 1.2, "method": "splice"},
            {"effect": "gain", "gain_db": 8.979},
            {"effect": "noise", "level_db": -50.79},
            {"effect": "lowpass", "frequency_hz": 7600, "q": 0.5098},
            {"effect": "lowpass", "frequency_hz": 7600, "q": 0.6013},
            {"effect": "lowpass", "frequency_hz": 7600, "q": 0.9},
            {"effect": "lowpass", "frequency_hz": 7600, "q": 2.5629},
        ]])
    },
};

const LIZARD_TONGUE: Known = Known {
    filter: "[0:a] asplit [out0][out2]; [out0] asetrate=%SAMPLE_RATE%*0.9,\
             aresample=%SAMPLE_RATE%,atempo=1/0.9,aformat=channel_layouts=mono,volume=0.2 \
             [p0]; [out2] asetrate=%SAMPLE_RATE%*1.1,aresample=%SAMPLE_RATE%,atempo=1/1.1,\
             aformat=channel_layouts=mono,volume=0.2[p2]; [p0][0][p2] amix=inputs=3",
    effects: |_| json!([three_tongues(-1.824, 1.65, 0.0667)]),
};

const ALIEN_TONGUE: Known = Known {
    filter: "[0:a] asplit [out0][out2]; [out0] asetrate=%SAMPLE_RATE%*0.8,\
             aresample=%SAMPLE_RATE%,atempo=1/0.8,aformat=channel_layouts=mono [p0]; [out2] \
             asetrate=%SAMPLE_RATE%*1.2,aresample=%SAMPLE_RATE%,atempo=1/1.2,\
             aformat=channel_layouts=mono[p2]; [p0][0][p2] amix=inputs=3",
    effects: |_| json!([three_tongues(-3.863, 3.156, 0.3333)]),
};

const ETHEREAL_TONGUE: Known = Known {
    filter: "[0:a] asplit [out0][out2]; [out0] asetrate=%SAMPLE_RATE%*0.99,\
             aresample=%SAMPLE_RATE%,volume=0.3 [p0]; [p0][out2] amix=inputs=2",
    effects: |_| {
        json!([{
            "nodes": {
                "slow": {"effect": "speed", "ratio": 0.99},
                "mix": {"effect": "mix", "weights": {"slow": 0.15, "input": 0.5}},
            },
            "edges": [["input", "slow"], ["input", "mix"], ["slow", "mix"], ["mix", "output"]],
        }])
    },
};

const SNAIL_TONGUE: Known = Known {
    filter: "atempo=0.5",
    effects: |_| json!([[{"effect": "tempo", "ratio": 0.5, "method": "splice"}]]),
};

const STUTTER: Known = Known {
    filter: "tremolo=f=10:d=0.8,rubberband=tempo=0.5",
    effects: |_| {
        json!([[
            {"effect": "tremolo", "rate_hz": 10, "depth": 0.8},
            {"effect": "tempo", "ratio": 0.5},
        ]])
    },
};

const CULT_SLURRING: Known = Known {
    filter: "rubberband=pitch=0.5,vibrato=5",
    effects: |_| {
        json!([[
            {"effect": "pitch", "semitones": -12},
            {"effect": "vibrato", "rate_hz": 5, "depth_semitones": 0.74},
        ]])
    },
};

const POLY: Known = Known {
    filter: "rubberband=pitch=1.5",
    effects: |_| json!([[{"effect": "pitch", "semitones": 7.02}]]),
};

const FISH_TONGUE: Known = Known {
    filter: "rubberband=pitch='ifnot(%BLIPS%,2-(%PITCH%+if(%FEMALE%,4))/16,1)':\
             formant=preserved,highpass=f=1000:t=s:w=24,equalizer=f=1200:g=15,\
             equalizer=f=4350:g=-15,highshelf=f=870:g=1,\
             afftfilt=real='st(0,(b+0.5)/nb*sr);st(1,3000+1500*sin(9.3*2*PI*pts));\
             st(2,ld(0)/ld(1));re*(1-ld(2)^2+2*gauss(log(ld(2)+1)))':\
             imag='st(0,(b+0.5)/nb*sr);st(1,3000+1500*sin(9.3*2*PI*pts));\
             st(2,ld(0)/ld(1));im*(1-ld(2)^2+2*gauss(log(ld(2)+1)))':win_size=1024",
    effects: fish_tongue,
};

const KNOWN: [Known; 15] = [
    SYNTHETIC,
    TRAM,
    FISHING_CAPSULE,
    GAS_MASK,
    MEGAPHONE,
    MEGAPHONE_EMAGGED,
    SECHAILER,
    LIZARD_TONGUE,
    ALIEN_TONGUE,
    ETHEREAL_TONGUE,
    SNAIL_TONGUE,
    STUTTER,
    CULT_SLURRING,
    POLY,
    FISH_TONGUE,
];

/// The machines' voice.
fn synthetic() -> Value {
    json!([
        {"effect": "gain", "gain_db": -0.915},
        {"effect": "limiter", "ceiling_db": 0},
        {"effect": "compressor", "threshold_db": -13.979, "ratio": 20, "attack_ms": 10,
            "release_ms": 50, "makeup_db": 6.021, "knee_db": 9.031},
        {"effect": "highpass", "frequency_hz": 1000},
    ])
}

/// The voice beside itself shifted down by `low` and up by `high` semitones, each at `weight`.
fn three_tongues(low: f64, high: f64, weight: f64) -> Value {
    json!({
        "nodes": {
            "low": {"effect": "pitch", "semitones": low, "method": "splice"},
            "high": {"effect": "pitch", "semitones": high, "method": "splice"},
            "mix": {"effect": "mix", "weights": {"low": weight, "input": 0.3333, "high": weight}},
        },
        "edges": [
            ["input", "low"],
            ["input", "high"],
            ["input", "mix"],
            ["low", "mix"],
            ["high", "mix"],
            ["mix", "output"],
        ],
    })
}

/// The fish tongue's voice: its speech pitched up with the formants kept, less for a higher
/// voice, by the ratio its rubberband is given, and its blips not at all.
fn fish_tongue(filled: Filled) -> Value {
    let pitch = (!filled.blips).then(|| {
        let female = if filled.female { 4.0 } else { 0.0 };
        let ratio: f64 = 2.0 - (filled.pitch + female) / 16.0;
        json!({"effect": "pitch", "semitones": 12.0 * ratio.log2(), "keep_formants": true})
    });
    let rest = [
        json!({"effect": "highpass", "frequency_hz": 1000, "q": 3.464}),
        json!({"effect": "peak", "frequency_hz": 1200, "gain_db": 15, "q": 1}),
        json!({"effect": "peak", "frequency_hz": 4350, "gain_db": -15, "q": 1}),
        json!({"effect": "highshelf", "frequency_hz": 870, "gain_db": 1, "q": 0.5}),
        json!({"effect": "wobble", "center_hz": 1500, "swing_hz": 750, "rate_hz": 37.5}),
    ];
    json!([pitch.into_iter().chain(rest).collect::<Vec<_>>()])
}

impl Known {
    /// How long a start of `graph` this filter is, and the numbers it has in its placeholders.
    fn matched(&self, graph: &str) -> Option<(usize, Filled)> {
        let (mut pattern, mut rest) = (self.filter, graph);
        let mut filled = Filled::default();
        loop {
            let next = (PLACEHOLDERS.iter())
                .filter_map(|&placeholder| Some((pattern.find(placeholder)?, placeholder)))
                .min();
            let Some((at, placeholder)) = next else {
                let after = rest.strip_prefix(pattern)?;
                return Some((graph.len() - after.len(), filled));
            };
            rest = rest.strip_prefix(&pattern[..at])?;
            match placeholder {
                "%PITCH%" => (filled.pitch, rest) = number(rest)?,
                "%FEMALE%" => (filled.female, rest) = flag(rest)?,
                _ => (filled.blips, rest) = flag(rest)?,
            }
            pattern = &pattern[at + placeholder.len()..];
        }
    }
}

/// The truth `text` starts with, as DM writes one, 1 or 0, and the rest of it.
fn flag(text: &str) -> Option<(bool, &str)> {
    match text.strip_prefix('1') {
        Some(after) => Some((true, after)),
        None => Some((false, text.strip_prefix('0')?)),
    }
}

/// The number `text` starts with, as DM writes one, and the rest of it.
fn number(text: &str) -> Option<(f64, &str)> {
    let digits = |s: &str| s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let sign = usize::from(text.starts_with('-'));
    let mut end = sign + digits(&text[sign..]);
    if end == sign {
        return None;
    }
    if let Some(fraction) = text[end..].strip_prefix('.') {
        let more = digits(fraction);
        if more > 0 {
            end += 1 + more;
        }
    }
    Some((text[..end].parse().ok()?, &text[end..]))
}

/// The effects in place of `graph`, a filter graph from the game stripped of its double quotes,
/// if it is made only of known filters joined by commas: theirs, in its order.
pub fn pipeline(graph: &str, impulses: &Impulses) -> Option<Pipeline> {
    Pipeline::new(&effects(graph)?, impulses).ok()
}

/// The graphs of [`pipeline`].
fn effects(graph: &str) -> Option<Vec<Graph>> {
    KNOWN.iter().find_map(|known| {
        let (length, filled) = known.matched(graph)?;
        let mut graphs = schema::parse(&(known.effects)(filled)).ok()?;
        match &graph[length..] {
            "" => {}
            rest => graphs.extend(effects(rest.strip_prefix(',')?)?),
        }
        Some(graphs)
    })
}

/// The filter graphs ffmpeg has been given, so that each is logged as it first comes: what the
/// game sends that has no effects in its place.
#[derive(Debug, Default)]
pub struct Unknown(Mutex<HashSet<String>>);

impl Unknown {
    /// Notes that ffmpeg runs `graph`, logging it if it has not before.
    pub fn note(&self, graph: &str) {
        let mut seen = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(graph) {
            seen.insert(graph.to_owned());
            warn!("running ffmpeg on a filter graph with no effects in its place: {graph}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, io, sync::Arc};

    use effects::schema::Node;

    use super::*;

    /// Where fixtures/effects.py runs a filter otherwise than the game sends it: the hailer's
    /// noise seeded, so that a rerun writes the same bytes, and the fishing spot capsule's
    /// compressor at the most ratio ffmpeg takes.
    const RAN_AS: [(&str, &str); 2] = [
        ("anoisesrc=a=0.01:d=60 ", "anoisesrc=a=0.01:d=60:seed=1 "),
        (":ratio=40:", ":ratio=20:"),
    ];

    /// The filters fixtures/effects.py ran through ffmpeg, by the name of its case, and the
    /// effects it compared with what ffmpeg made of each.
    fn compared() -> BTreeMap<String, (String, Vec<Graph>)> {
        let fixture = testkit::Fixture::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../effects/tests/fixtures/effects.safetensors"
        ));
        let cases: Vec<Value> = serde_json::from_str(&fixture.metadata()["cases"]).unwrap();
        (cases.into_iter())
            .map(|case| {
                let name = case["name"].as_str().unwrap().to_owned();
                let ffmpeg = case["ffmpeg"].as_str().unwrap().to_owned();
                (name, (ffmpeg, schema::parse(&case["effects"]).unwrap()))
            })
            .collect()
    }

    /// `known`'s filter as the game sends it with pitch disabled, for a man's voice's speech, as
    /// fixtures/effects.py filled it.
    fn sent(known: &Known) -> String {
        filled(known, "0", "0", "0")
    }

    /// `known`'s filter as the game sends it with this pitch, female and blips.
    fn filled(known: &Known, pitch: &str, female: &str, blips: &str) -> String {
        (known.filter)
            .replace("%PITCH%", pitch)
            .replace("%FEMALE%", female)
            .replace("%BLIPS%", blips)
    }

    #[test]
    fn every_filter_is_what_was_compared_with_ffmpeg() {
        let compared = compared();
        let mut cases = HashSet::new();
        let mut ran_as = [0; RAN_AS.len()];
        for known in &KNOWN {
            let sent = sent(known);
            let mut run = sent.replace("%SAMPLE_RATE%", "48000");
            for ((game, fixture), n) in RAN_AS.iter().zip(&mut ran_as) {
                if run.contains(game) {
                    run = run.replace(game, fixture);
                    *n += 1;
                }
            }
            let (case, (_, effects_compared)) = (compared.iter())
                .find(|(_, (ran, _))| *ran == run)
                .unwrap_or_else(|| panic!("no case ran {run}"));
            assert!(cases.insert(case), "{case} twice");
            assert_eq!(effects(&sent).as_ref(), Some(effects_compared), "{case}");
        }
        assert_eq!(ran_as, [1; RAN_AS.len()]);
    }

    #[test]
    fn splits_what_the_game_joins() {
        let one = |known: &Known| effects(&sent(known)).unwrap();
        // In the order the game joins them: the speaker's own, the status effects', a
        // megaphone's, the mask's.
        let joined = [
            &[STUTTER, GAS_MASK][..],
            &[SYNTHETIC, CULT_SLURRING, GAS_MASK],
            &[LIZARD_TONGUE, MEGAPHONE],
            &[POLY, MEGAPHONE_EMAGGED],
            &[MEGAPHONE, SECHAILER],
            &[
                ALIEN_TONGUE,
                STUTTER,
                CULT_SLURRING,
                MEGAPHONE_EMAGGED,
                SECHAILER,
            ],
            &[FISH_TONGUE, STUTTER, GAS_MASK],
            &[TRAM, TRAM],
        ];
        for filters in joined {
            let graph: Vec<String> = filters.iter().map(sent).collect();
            let graph = graph.join(",");
            let each: Vec<Graph> = filters.iter().flat_map(one).collect();
            assert_eq!(effects(&graph), Some(each), "{graph}");
        }
    }

    #[test]
    fn pitches_the_fish_tongue_as_the_game_asks() {
        let mask = sent(&GAS_MASK);
        let pitch = |graph: &str| {
            let graphs = effects(graph).unwrap();
            let (Graph::Chain(fish), Graph::Chain(mask)) = (&graphs[0], &graphs[1]) else {
                panic!("{graphs:?}");
            };
            let semitones = match fish[0] {
                Node::Pitch {
                    semitones,
                    keep_formants: true,
                    ..
                } => Some(semitones),
                Node::Highpass { .. } => None,
                ref node => panic!("{node:?}"),
            };
            let pitched = usize::from(semitones.is_some());
            assert_eq!((fish.len(), mask.len()), (5 + pitched, 2), "{graphs:?}");
            semitones
        };
        // 12 log2(2 - (pitch + 4 for a woman) / 16), and none for blips.
        for (placeholders, semitones) in [
            (["0", "0", "0"], Some(12.0)),
            (["5", "0", "0"], Some(9.0587)),
            (["-12", "1", "0"], Some(15.8631)),
            (["12", "1", "0"], Some(0.0)),
            (["7", "0", "1"], None),
        ] {
            let [p, female, blips] = placeholders;
            let fish = filled(&FISH_TONGUE, p, female, blips);
            let found = pitch(&format!("{fish},{mask}"));
            let near = match (found, semitones) {
                (Some(found), Some(semitones)) => (found - semitones).abs() < 1e-4,
                (found, semitones) => found == semitones,
            };
            assert!(near, "{placeholders:?}: {found:?}");
        }
    }

    #[test]
    fn leaves_to_ffmpeg_what_it_does_not_know() {
        let impulses = Impulses {
            synth: Arc::new(effects::convolve::Response::new(&[1.0, 1.0], 2, dsp::RATE)),
        };
        let gas_mask = sent(&GAS_MASK);
        assert!(pipeline(&gas_mask, &impulses).is_some());
        for graph in [
            String::new(),
            "volume=0.5".to_owned(),
            format!("{gas_mask},"),
            format!("{gas_mask}.5"),
            format!("{gas_mask} ,{gas_mask}"),
            format!("{},volume=0.5,{gas_mask}", sent(&STUTTER)),
            FISH_TONGUE.filter.to_owned(),
            filled(&FISH_TONGUE, "x", "0", "0"),
            filled(&FISH_TONGUE, "0", "2", "0"),
            // Pitched down past what rubberband takes, and past what the effects do.
            filled(&FISH_TONGUE, "40", "0", "0"),
            filled(&FISH_TONGUE, "30", "0", "0"),
            // TerraGov Marine Corps's silicons and combat robots.
            "afftfilt=real='hypot(re,im)*sin(0)':imag='hypot(re,im)*cos(0)':win_size=512:\
             overlap=1,rubberband=pitch=0.8"
                .to_owned(),
        ] {
            assert!(pipeline(&graph, &impulses).is_none(), "{graph}");
        }
    }

    /// What is logged while `f` runs on this thread.
    fn logs(f: impl FnOnce()) -> String {
        #[derive(Clone, Default)]
        struct Buffer(Arc<Mutex<Vec<u8>>>);

        impl io::Write for Buffer {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let buffer = Buffer::default();
        let writer = buffer.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn logs_each_unknown_graph_once() {
        let unknown = Unknown::default();
        let logged = logs(|| {
            for graph in ["volume=0.5", "atempo=2", "volume=0.5", "volume=0.5"] {
                unknown.note(graph);
            }
        });
        let lines: Vec<&str> = logged.lines().collect();
        assert_eq!(lines.len(), 2, "{logged}");
        assert!(lines[0].contains("WARN") && lines[0].ends_with(": volume=0.5"));
        assert!(lines[1].ends_with(": atempo=2"));
    }
}
