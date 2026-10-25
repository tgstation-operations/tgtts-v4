//! The radio handoff: the speech and the blips of a message, by its identifier, for the radio
//! routes to put on the radio.
//!
//! The game asks for all five versions of a message at once, so a radio route waits for what it
//! is to put on the radio, up to a bound; the old services polled for a file under the
//! identifier, here the routes hand the audio over in memory. What was handed over is kept for
//! as long as a radio route would wait for it.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use tokio::{sync::watch, time::Instant};

use crate::wav::Audio;

/// Whose audio a radio route puts on the radio.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Source {
    /// `/tts`'s, for `/tts-radio`.
    Speech,
    /// `/tts-blips`'s, for `/tts-blips-radio`.
    Blips,
}

/// Why a radio route got nothing to put on the radio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Missing {
    /// Nothing was handed over within the bound.
    TimedOut,
    /// The route that was to hand it over failed with this status.
    Failed(u16),
}

#[derive(Clone, Debug)]
enum State {
    Pending,
    Ready(Arc<Audio>),
    Failed(u16),
}

#[derive(Debug)]
struct Slot {
    state: watch::Sender<State>,
    touched: Instant,
}

/// Audio handed from the routes that make it to the radio routes.
#[derive(Debug)]
pub struct Handoff {
    bound: Duration,
    slots: Mutex<HashMap<(Source, String), Slot>>,
}

impl Handoff {
    /// A handoff where a radio route waits up to `bound`, and what is handed over is kept as
    /// long.
    pub fn new(bound: Duration) -> Self {
        Self {
            bound,
            slots: Mutex::default(),
        }
    }

    /// Hands over `source`'s audio for `identifier`, or the status its route failed with,
    /// replacing what was handed over for it before.
    pub fn publish(&self, source: Source, identifier: &str, audio: Result<Arc<Audio>, u16>) {
        let state = match audio {
            Ok(audio) => State::Ready(audio),
            Err(status) => State::Failed(status),
        };
        self.slot(source, identifier, |slot| {
            slot.state.send_replace(state);
        });
    }

    /// `source`'s audio for `identifier`, once it is handed over.
    pub async fn wait(&self, source: Source, identifier: &str) -> Result<Arc<Audio>, Missing> {
        let mut state = self.slot(source, identifier, |slot| slot.state.subscribe());
        let ready = state.wait_for(|state| !matches!(state, State::Pending));
        match tokio::time::timeout(self.bound, ready).await {
            Ok(Ok(state)) => match &*state {
                State::Ready(audio) => Ok(audio.clone()),
                State::Failed(status) => Err(Missing::Failed(*status)),
                State::Pending => unreachable!("waited for a state that is not pending"),
            },
            // A slot is only dropped once it has gone untouched for the bound, so only a wait
            // that is timing out anyway can see it closed.
            Ok(Err(_)) | Err(_) => Err(Missing::TimedOut),
        }
    }

    /// Calls `f` on the slot of `source` and `identifier`, made if there is none, having dropped
    /// every slot untouched for longer than the bound.
    fn slot<T>(&self, source: Source, identifier: &str, f: impl FnOnce(&mut Slot) -> T) -> T {
        let now = Instant::now();
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        slots.retain(|_, slot| now.duration_since(slot.touched) < self.bound);
        let slot = slots
            .entry((source, identifier.to_owned()))
            .or_insert_with(|| Slot {
                state: watch::Sender::new(State::Pending),
                touched: now,
            });
        slot.touched = now;
        f(slot)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.slots.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        pin::pin,
        task::{Context, Poll, Waker},
    };

    use super::*;

    const LONG: Duration = Duration::from_secs(3600);

    fn audio(n: f32) -> Arc<Audio> {
        Arc::new(Audio {
            rate: 48000,
            channels: 1,
            samples: vec![n],
        })
    }

    fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        future.poll(&mut Context::from_waker(Waker::noop()))
    }

    #[tokio::test]
    async fn waits_for_what_is_handed_over() {
        let handoff = Handoff::new(LONG);
        let mut wait = pin!(handoff.wait(Source::Speech, "id"));
        assert!(poll_once(wait.as_mut()).is_pending());
        handoff.publish(Source::Speech, "id", Ok(audio(1.0)));
        assert_eq!(wait.await, Ok(audio(1.0)));
    }

    #[tokio::test]
    async fn gives_what_was_handed_over_before() {
        let handoff = Handoff::new(LONG);
        handoff.publish(Source::Blips, "id", Ok(audio(1.0)));
        handoff.publish(Source::Speech, "id", Ok(audio(2.0)));
        assert_eq!(handoff.wait(Source::Blips, "id").await, Ok(audio(1.0)));
        assert_eq!(handoff.wait(Source::Speech, "id").await, Ok(audio(2.0)));
        handoff.publish(Source::Speech, "id", Ok(audio(3.0)));
        assert_eq!(handoff.wait(Source::Speech, "id").await, Ok(audio(3.0)));
    }

    #[tokio::test]
    async fn every_waiter_gets_it() {
        let handoff = Handoff::new(LONG);
        let mut first = pin!(handoff.wait(Source::Speech, "id"));
        let mut second = pin!(handoff.wait(Source::Speech, "id"));
        assert!(poll_once(first.as_mut()).is_pending());
        assert!(poll_once(second.as_mut()).is_pending());
        handoff.publish(Source::Speech, "id", Ok(audio(1.0)));
        assert_eq!(first.await, Ok(audio(1.0)));
        assert_eq!(second.await, Ok(audio(1.0)));
    }

    #[tokio::test]
    async fn passes_a_failure_on() {
        let handoff = Handoff::new(LONG);
        let mut wait = pin!(handoff.wait(Source::Blips, "id"));
        assert!(poll_once(wait.as_mut()).is_pending());
        handoff.publish(Source::Blips, "id", Err(500));
        assert_eq!(wait.await, Err(Missing::Failed(500)));
    }

    #[tokio::test]
    async fn times_out() {
        let handoff = Handoff::new(Duration::from_millis(1));
        handoff.publish(Source::Blips, "id", Ok(audio(1.0)));
        assert_eq!(
            handoff.wait(Source::Speech, "id").await,
            Err(Missing::TimedOut)
        );
        assert_eq!(
            handoff.wait(Source::Blips, "other").await,
            Err(Missing::TimedOut)
        );
    }

    #[tokio::test]
    async fn keeps_slots_only_as_long_as_the_bound() {
        let handoff = Handoff::new(Duration::ZERO);
        handoff.publish(Source::Speech, "first", Ok(audio(1.0)));
        handoff.publish(Source::Speech, "second", Ok(audio(2.0)));
        assert_eq!(handoff.len(), 1);
        let handoff = Handoff::new(LONG);
        handoff.publish(Source::Speech, "first", Ok(audio(1.0)));
        handoff.publish(Source::Speech, "second", Ok(audio(2.0)));
        assert_eq!(handoff.len(), 2);
    }
}
