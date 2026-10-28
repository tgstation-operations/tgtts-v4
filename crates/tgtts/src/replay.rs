//! A stream's bytes kept for whoever reads it: each reader has them from the start, then each
//! run as it comes, until the stream ends, however late it starts reading.

use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

use axum::body::Bytes;
use futures_util::{Stream, stream};
use tokio::sync::watch;

/// The bytes of a stream so far, and whether it has ended.
#[derive(Debug)]
pub struct Replay {
    state: Mutex<State>,
    changed: watch::Sender<()>,
}

#[derive(Debug, Default)]
struct State {
    runs: Vec<Bytes>,
    ended: bool,
}

impl Default for Replay {
    fn default() -> Self {
        Self {
            state: Mutex::default(),
            changed: watch::Sender::new(()),
        }
    }
}

impl Replay {
    /// Adds `bytes` to the stream, unless it has ended.
    pub fn push(&self, bytes: Vec<u8>) {
        if bytes.is_empty() {
            return;
        }
        let mut state = self.state();
        if !state.ended {
            state.runs.push(Bytes::from(bytes));
            drop(state);
            self.changed.send_replace(());
        }
    }

    /// Ends the stream.
    pub fn end(&self) {
        self.state().ended = true;
        self.changed.send_replace(());
    }

    /// Whether the stream has ended.
    pub fn ended(&self) -> bool {
        self.state().ended
    }

    /// The bytes so far.
    pub fn bytes(&self) -> Vec<u8> {
        self.state().runs.concat()
    }

    /// The stream from its start, each run as it comes, ending as it does.
    pub fn reader(self: &Arc<Self>) -> impl Stream<Item = Result<Bytes, Infallible>> + use<> {
        let reading = (self.clone(), 0, self.changed.subscribe());
        stream::unfold(reading, |(replay, mut read, mut changed)| async move {
            loop {
                // Marked seen before the state is read, so that a run pushed after is waited
                // for rather than missed.
                changed.borrow_and_update();
                {
                    let state = replay.state();
                    if read < state.runs.len() {
                        let bytes = Bytes::from(state.runs[read..].concat());
                        read = state.runs.len();
                        drop(state);
                        return Some((Ok(bytes), (replay, read, changed)));
                    }
                    if state.ended {
                        return None;
                    }
                }
                changed.changed().await.ok()?;
            }
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;

    #[tokio::test]
    async fn a_reader_has_it_all_however_late_it_starts() {
        let replay = Arc::new(Replay::default());
        replay.push(b"one ".to_vec());
        let mut early = Box::pin(replay.reader());
        assert_eq!(early.next().await.unwrap().unwrap(), "one ");
        replay.push(b"two ".to_vec());
        replay.push(Vec::new());
        let waiting = tokio::spawn(async move {
            let mut read = Vec::new();
            while let Some(bytes) = early.next().await {
                read.extend_from_slice(&bytes.unwrap());
            }
            read
        });
        tokio::task::yield_now().await;
        replay.push(b"three".to_vec());
        replay.end();
        replay.push(b" after".to_vec());
        assert_eq!(waiting.await.unwrap(), b"two three");
        let late: Vec<Bytes> = replay.reader().map(Result::unwrap).collect().await;
        assert_eq!(late.concat(), b"one two three");
        assert_eq!(replay.bytes(), b"one two three");
        assert!(replay.ended());
    }
}
