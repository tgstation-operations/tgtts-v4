//! What the HTTP tests share: the service on a port of its own, and the game's requests to it,
//! made through ureq as rust-g makes them.

#![allow(dead_code)]

pub mod real;

use std::{
    io::Write,
    process::{Command, Stdio},
    sync::Arc,
    thread,
    time::Duration,
};

use tgtts::{
    service::{self, Service, Setup},
    wav::{self, Audio},
};

pub const TOKEN: &str = "s3cret-token";
pub const VOICE: &str = "Man (Test)";

/// The service, served on a port of its own.
pub struct Server {
    pub runtime: tokio::runtime::Runtime,
    pub base: String,
    pub service: Arc<Service>,
}

/// Serves the service `setup` makes, with [`TOKEN`] as its token.
pub fn serve(setup: Setup) -> Server {
    let service = Service::new(Setup {
        token: TOKEN.to_owned(),
        ..setup
    });
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let router = service::router(service.clone());
    runtime.spawn(async move { axum::serve(listener, router).await });
    Server {
        runtime,
        base,
        service,
    }
}

impl Server {
    pub fn get(&self, path: &str, body: &str) -> Reply {
        call(&format!("{}{path}", self.base), Some(TOKEN), body)
    }

    /// A POST with the token, its body JSON.
    pub fn post(&self, path: &str, body: &str) -> Reply {
        let request = ureq::post(&format!("{}{path}", self.base))
            .set("Authorization", TOKEN)
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(60));
        reply(request.send_bytes(body.as_bytes()), path)
    }
}

#[derive(Debug)]
pub struct Reply {
    pub status: u16,
    pub content_type: Option<String>,
    pub disposition: Option<String>,
    /// The CORS header that lets a page of any origin read it.
    pub allow_origin: Option<String>,
    pub length: Option<String>,
    pub body: Vec<u8>,
}

impl Reply {
    /// The length the game takes from `audio-length`, in frames at `rate`.
    pub fn frames(&self, rate: u32) -> usize {
        let seconds: f64 = self.length.as_deref().unwrap().parse().unwrap();
        (seconds * f64::from(rate)).round() as usize
    }

    pub fn text(&self) -> &str {
        std::str::from_utf8(&self.body).unwrap()
    }

    #[track_caller]
    pub fn assert_ogg(&self, rate: u32) {
        assert_eq!(self.status, 200, "{}", String::from_utf8_lossy(&self.body));
        assert_eq!(self.content_type.as_deref(), Some("audio/ogg"));
        assert_eq!(
            self.disposition.as_deref(),
            Some("attachment; filename=identifier.ogg")
        );
        let (_, header_rate, bitrate) = vorbis_header(&self.body);
        assert_eq!((header_rate, bitrate), (rate, 64000));
    }
}

/// The Vorbis identification header's channels, rate and nominal bitrate.
pub fn vorbis_header(ogg: &[u8]) -> (u8, u32, i32) {
    assert_eq!(&ogg[..4], b"OggS");
    let at = ogg.windows(7).position(|w| w == b"\x01vorbis").unwrap();
    let header = &ogg[at + 7..];
    let rate = u32::from_le_bytes(header[5..9].try_into().unwrap());
    let bitrate = i32::from_le_bytes(header[13..17].try_into().unwrap());
    (header[4], rate, bitrate)
}

/// A GET as rust-g's `http_request_async` sends it: the game's headers and its body.
pub fn call(url: &str, token: Option<&str>, body: &str) -> Reply {
    let mut request = ureq::get(url)
        .set("User-Agent", "rust-g/7.0.0")
        .set("Content-Type", "application/json")
        .timeout(Duration::from_secs(60));
    if let Some(token) = token {
        request = request.set("Authorization", token);
    }
    reply(request.send_bytes(body.as_bytes()), url)
}

fn reply(response: Result<ureq::Response, ureq::Error>, url: &str) -> Reply {
    let response = match response {
        Ok(response) | Err(ureq::Error::Status(_, response)) => response,
        Err(e) => panic!("{url}: {e}"),
    };
    let header = |name| response.header(name).map(str::to_owned);
    let (content_type, disposition) = (header("content-type"), header("content-disposition"));
    let allow_origin = header("access-control-allow-origin");
    let (status, length) = (response.status(), header("audio-length"));
    let mut body = Vec::new();
    std::io::Read::read_to_end(&mut response.into_reader(), &mut body).unwrap();
    Reply {
        status,
        content_type,
        disposition,
        allow_origin,
        length,
        body,
    }
}

/// BYOND's `url_encode`, as `tts_filter_encode` and the special filters go through it.
pub fn url_encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                char::from(b).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// A message as `queue_tts_message` sends it.
#[derive(Clone)]
pub struct Message<'a> {
    pub speaker: &'a str,
    pub identifier: &'a str,
    pub filter: &'a str,
    pub pitch: &'a str,
    pub special_filters: &'a str,
    pub blip_base: &'a str,
    pub blip_number: &'a str,
    pub text: &'a str,
    /// The effects as JSON, none given where empty.
    pub effects: &'a str,
}

impl Default for Message<'_> {
    fn default() -> Self {
        Self {
            speaker: VOICE,
            identifier: "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12.1234.5",
            filter: "",
            pitch: "0",
            special_filters: "",
            blip_base: "male",
            blip_number: "1",
            text: "Hello there.",
            effects: "",
        }
    }
}

impl Message<'_> {
    /// tts.dm's query for `route`, the speaker as it is, unencoded.
    pub fn query(&self, route: &str) -> String {
        let mut query = format!(
            "/{route}?voice={}&identifier={}&filter={}&pitch={}&special_filters={}",
            self.speaker,
            self.identifier,
            url_encode(self.filter),
            self.pitch,
            url_encode(self.special_filters)
        );
        if route.starts_with("tts-blips") {
            query += &format!(
                "&blip_base={}&blip_number={}",
                self.blip_base, self.blip_number
            );
        }
        query
    }

    pub fn body(&self) -> String {
        self.with_effects(serde_json::json!({ "text": self.text }))
    }

    /// The body of `POST /stream` for the routes `variants`.
    pub fn job(&self, variants: &[&str]) -> String {
        let mut body = serde_json::json!({
            "voice": self.speaker,
            "identifier": self.identifier,
            "text": self.text,
            "pitch": self.pitch.parse::<i32>().unwrap(),
            "filter": self.filter,
            "special_filters": self.special_filters,
            "blip_base": self.blip_base,
            "blip_number": self.blip_number,
            "variants": variants,
        });
        if !self.effects.is_empty() {
            body["effects"] = serde_json::from_str(self.effects).unwrap();
        }
        body.to_string()
    }

    /// The gibberish request's body, which the game sends with the same text twice.
    pub fn gibberish_body(&self) -> String {
        self.with_effects(serde_json::json!({ "raw_text": self.text, "gibberish_text": self.text }))
    }

    fn with_effects(&self, mut body: serde_json::Value) -> String {
        if !self.effects.is_empty() {
            body["effects"] = serde_json::from_str(self.effects).unwrap();
        }
        body.to_string()
    }

    pub fn speech(&self, server: &Server) -> Reply {
        server.get(&self.query("tts"), &self.body())
    }

    pub fn blips(&self, server: &Server) -> Reply {
        server.get(&self.query("tts-blips"), &self.body())
    }

    pub fn radio(&self, server: &Server) -> Reply {
        server.get(&self.query("tts-radio"), &self.body())
    }

    pub fn blips_radio(&self, server: &Server) -> Reply {
        server.get(&self.query("tts-blips-radio"), &self.body())
    }

    pub fn gibberish(&self, server: &Server) -> Reply {
        server.get(&self.query("tts-radio"), &self.gibberish_body())
    }

    /// All five, at once, as `start_requests` sends a message that is not local.
    pub fn all(&self, server: &Server) -> [Reply; 5] {
        thread::scope(|s| {
            let calls = [
                s.spawn(|| self.speech(server)),
                s.spawn(|| self.blips(server)),
                s.spawn(|| self.radio(server)),
                s.spawn(|| self.blips_radio(server)),
                s.spawn(|| self.gibberish(server)),
            ];
            calls.map(|call| call.join().unwrap())
        })
    }
}

/// The length `frames` take resampled from `from` Hz to `to`, as dsp's resampling gives it.
pub fn resampled(frames: usize, from: u32, to: u32) -> usize {
    (frames as f64 * (f64::from(to) / f64::from(from))).ceil() as usize
}

/// The length of `frames` of speech at 48 kHz over the radio: resampled to 8 kHz and back,
/// between clicks of `on` and `off` frames at 48 kHz, with 0.1 s of static before the second.
pub fn radio_frames(frames: usize, on: usize, off: usize) -> usize {
    on + resampled(resampled(frames, 48000, 8000), 8000, 48000) + 4800 + off
}

/// The codec, rate, channels and duration ffprobe reads of the Ogg `ogg`, from a file, since from
/// a pipe it cannot find the end to time it by.
pub fn ffprobe(ogg: &[u8]) -> (String, u32, u16, f64) {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), ogg).unwrap();
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-of", "json", "-show_entries"])
        .args(["stream=codec_name,sample_rate,channels:format=duration"])
        .arg(file.path())
        .output()
        .expect("ffprobe runs");
    assert!(output.status.success(), "ffprobe could not read the Ogg");
    let probe: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let stream = &probe["streams"][0];
    let number = |value: &serde_json::Value| value.as_str().unwrap().parse::<f64>().unwrap();
    (
        stream["codec_name"].as_str().unwrap().to_owned(),
        number(&stream["sample_rate"]) as u32,
        stream["channels"].as_u64().unwrap() as u16,
        number(&probe["format"]["duration"]),
    )
}

/// The Ogg `ogg` decoded by ffmpeg to float samples.
pub fn decode(ogg: &[u8]) -> Audio {
    let args = [
        "-f",
        "ogg",
        "-i",
        "pipe:0",
        "-c:a",
        "pcm_f32le",
        "-f",
        "wav",
        "pipe:1",
    ];
    let mut child = Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let (mut stdin, ogg) = (child.stdin.take().unwrap(), ogg.to_vec());
    let writer = thread::spawn(move || stdin.write_all(&ogg));
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap().unwrap();
    assert!(output.status.success(), "ffmpeg could not decode the Ogg");
    wav::read(&output.stdout).unwrap()
}
