//! Times a running service's streams as a player gets them: `stream_timing BASE [--jobs N]
//! [--at-once N] [--voices N] [--radio] [--streamed-radio] [--whole]`, with the token in `TOKEN`.
//!
//! It makes `N` jobs of the lines below, each with its job's number and the run's start time, so
//! that no run asks for a line another asked for and none comes from the cache, in voices spread
//! over `/tts-voices`, or over `--voices` of them, so that the model keeps what it computed of
//! each, `--at-once` of them at a time, and plays each job's speech stream, and its radio with
//! `--radio`, as it comes, noting when each page of it arrives. For each stream it writes a line
//! of JSON: from the job's making to the first page of audio, `first_audio_ms`; to the stream's
//! end, `done_ms`; the audio's length; and `buffer_ms`, the least a player that starts playing
//! once the first audio has come and that much more has passed must hold to play it without a
//! gap: the most any page came after its audio was due. With `--whole` it asks the whole-file
//! route for its lines instead and times each whole reply, which is when the game can play it
//! today. It ends with the medians and the 90th percentiles.

use std::{
    env,
    io::Read,
    process::ExitCode,
    sync::{Mutex, mpsc},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use argh::FromArgs;
use tgtts::ogg;

/// The lines it asks for, from a word to a long sentence.
const LINES: [&str; 8] = [
    "Yes.",
    "Get away from me!",
    "Hello there, could someone please come to medbay? I need help.",
    "The captain wants everyone on the bridge in five minutes.",
    "I think the engine is about to blow, we need to leave right now, everyone get to the pods.",
    "Security, there is a man with a toolbox in the armory.",
    "Cargo, we need more plasma sheets for research, as soon as you can, and some glass too.",
    "What is going on out there?",
];

/// The samples a second of the streams.
const RATE: f64 = 48000.0;

/// Times a running service's streams as a player gets them, the token in TOKEN.
#[derive(FromArgs)]
struct Options {
    /// the service's address
    #[argh(positional, arg_name = "BASE")]
    base: String,
    /// the jobs made, 24 by default
    #[argh(option, arg_name = "N", default = "24", from_str_fn(count))]
    jobs: usize,
    /// the jobs made at a time, 1 by default
    #[argh(option, arg_name = "N", default = "1", from_str_fn(count))]
    at_once: usize,
    /// the voices the jobs are spread over, all by default
    #[argh(option, arg_name = "N", from_str_fn(count))]
    voices: Option<usize>,
    /// play each job's radio too
    #[argh(switch)]
    radio: bool,
    /// play each job's radio too, streamed as its speech is
    #[argh(switch)]
    streamed_radio: bool,
    /// time the whole-file route's replies instead
    #[argh(switch)]
    whole: bool,
}

/// `value` as a count of at least 1.
fn count(value: &str) -> Result<usize, String> {
    match value.parse() {
        Ok(0) => Err("not at least 1".to_owned()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

fn main() -> ExitCode {
    let mut options: Options = argh::from_env();
    options.radio |= options.streamed_radio;
    let token = env::var("TOKEN").unwrap_or_default();
    match run(&options, &token) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("stream_timing: {e}");
            ExitCode::FAILURE
        }
    }
}

type Failure = Box<dyn std::error::Error + Send + Sync>;

/// What a stream came to.
#[derive(Clone, Debug, Default)]
struct Timed {
    route: &'static str,
    first_audio_ms: f64,
    done_ms: f64,
    audio_seconds: f64,
    buffer_ms: f64,
}

fn run(options: &Options, token: &str) -> Result<(), Failure> {
    let all: Vec<String> = serde_json::from_str(
        &ureq::get(&format!("{}/tts-voices", options.base))
            .set("Authorization", token)
            .call()?
            .into_string()?,
    )?;
    let spread = options.voices.unwrap_or(all.len()).min(all.len());
    let voices: Vec<&str> = (0..spread)
        .map(|i| all[i * all.len() / spread].as_str())
        .collect();
    let run = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() % 100_000);
    let work: Mutex<Vec<usize>> = Mutex::new((0..options.jobs).rev().collect());
    let (timed, received) = mpsc::channel();
    thread::scope(|scope| {
        for _ in 0..options.at_once {
            let (work, timed, voices) = (&work, timed.clone(), &voices);
            scope.spawn(move || {
                while let Some(job) = work.lock().unwrap_or_else(|e| e.into_inner()).pop() {
                    let voice = voices[(job * 7919) % voices.len()];
                    let line = format!("{} {job}, {run}.", LINES[job % LINES.len()]);
                    let _ = timed.send((job, voice, one(options, token, job, voice, &line)));
                }
            });
        }
        drop(timed);
        let mut all: Vec<Timed> = Vec::new();
        for (job, voice, result) in received {
            match result {
                Ok(streams) => {
                    for t in streams {
                        let line = serde_json::json!({
                            "job": job,
                            "voice": voice,
                            "route": t.route,
                            "first_audio_ms": t.first_audio_ms,
                            "done_ms": t.done_ms,
                            "audio_seconds": t.audio_seconds,
                            "buffer_ms": t.buffer_ms,
                        });
                        println!("{line}");
                        all.push(t);
                    }
                }
                Err(e) => eprintln!("job {job} in {voice}: {e}"),
            }
        }
        for route in ["tts", "tts-radio"] {
            let of: Vec<&Timed> = all.iter().filter(|t| t.route == route).collect();
            if of.is_empty() {
                continue;
            }
            let stat = |f: fn(&Timed) -> f64| {
                let mut values: Vec<f64> = of.iter().map(|t| f(t)).collect();
                values.sort_by(f64::total_cmp);
                let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
                format!("{:.0} ({:.0})", at(0.5), at(0.9))
            };
            eprintln!(
                "{route}: {} streams, {} at once; median (p90) ms: first audio {}, done {}, \
                 buffer {}; audio {} s",
                of.len(),
                options.at_once,
                stat(|t| t.first_audio_ms),
                stat(|t| t.done_ms),
                stat(|t| t.buffer_ms),
                stat(|t| t.audio_seconds),
            );
        }
    });
    Ok(())
}

/// A job of `line` in `voice`, its streams played; or with `--whole`, the whole-file route's
/// reply timed.
fn one(
    options: &Options,
    token: &str,
    job: usize,
    voice: &str,
    line: &str,
) -> Result<Vec<Timed>, Failure> {
    let made = Instant::now();
    let identifier = format!("timing-{job}");
    if options.whole {
        let url = format!(
            "{}/tts?voice={}&identifier={identifier}",
            options.base,
            encode(voice)
        );
        let reply = ureq::get(&url)
            .set("Authorization", token)
            .set("Content-Type", "application/json")
            .send_string(&serde_json::json!({ "text": line }).to_string())?;
        let seconds: f64 = reply.header("audio-length").unwrap_or("0").parse()?;
        let mut body = Vec::new();
        reply.into_reader().read_to_end(&mut body)?;
        let ms = made.elapsed().as_secs_f64() * 1e3;
        return Ok(vec![Timed {
            route: "tts",
            first_audio_ms: ms,
            done_ms: ms,
            audio_seconds: seconds,
            buffer_ms: 0.0,
        }]);
    }
    let mut variants = vec!["tts"];
    if options.radio {
        variants.push("tts-radio");
    }
    let job = serde_json::json!({
        "voice": voice,
        "identifier": identifier,
        "text": line,
        "variants": variants,
        "streamed_radio": options.streamed_radio,
    });
    let reply: serde_json::Value = serde_json::from_str(
        &ureq::post(&format!("{}/stream", options.base))
            .set("Authorization", token)
            .set("Content-Type", "application/json")
            .send_string(&job.to_string())?
            .into_string()?,
    )?;
    thread::scope(|scope| {
        let played: Vec<_> = (variants.iter())
            .map(|&route| {
                let path = reply["streams"][route]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned();
                let url = format!("{}{path}", options.base);
                scope.spawn(move || play(&url, route, made))
            })
            .collect();
        played
            .into_iter()
            .map(|p| p.join().map_err(|_| Failure::from("a player panicked"))?)
            .collect()
    })
}

/// The stream at `url`, played as it comes, timed from `made`.
fn play(url: &str, route: &'static str, made: Instant) -> Result<Timed, Failure> {
    let mut reader = ureq::get(url).call()?.into_reader();
    let (mut body, mut buffer) = (Vec::new(), vec![0; 16384]);
    // When each run of bytes had come, and how many had in all.
    let mut arrivals: Vec<(Duration, usize)> = Vec::new();
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buffer[..n]);
        arrivals.push((made.elapsed(), body.len()));
    }
    let done_ms = made.elapsed().as_secs_f64() * 1e3;
    let pages = ogg::pages(&body)?;
    let head = ogg::packets(&pages[..1])
        .into_iter()
        .next()
        .filter(|p| p.starts_with(b"OpusHead") && p.len() >= 12)
        .ok_or("no OpusHead")?;
    let pre_skip = f64::from(u16::from_le_bytes([head[10], head[11]]));
    // Each page's end in the body, when it had all come, and the samples before it.
    let (mut end, mut before, mut first, mut late) = (0, 0.0f64, None, 0f64);
    for page in &pages {
        end += 27 + page.lacing.len() + page.body.len();
        let came = arrivals
            .iter()
            .find(|(_, total)| *total >= end)
            .map(|(at, _)| at.as_secs_f64())
            .ok_or("a page that never came")?;
        let Some(granule) = page.granule.map(|g| g as f64).filter(|&g| g > pre_skip) else {
            continue;
        };
        let first = *first.get_or_insert(came);
        let due = first + (before.max(pre_skip) - pre_skip) / RATE;
        late = late.max(came - due);
        before = granule;
    }
    let first = first.ok_or("no audio")?;
    Ok(Timed {
        route,
        first_audio_ms: first * 1e3,
        done_ms,
        audio_seconds: (before - pre_skip).max(0.0) / RATE,
        buffer_ms: late * 1e3,
    })
}

/// `text` as a query's value.
fn encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                char::from(b).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}
