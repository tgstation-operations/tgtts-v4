//! The radio handoff: the speech and the blips of a message, by its identifier, for the radio
//! routes to put on the radio.
//!
//! The game asks for all five versions of a message at once, so a radio route waits for what it
//! is to put on the radio; the old services polled for a file under the identifier, here the
//! routes hand the audio over in memory. A radio route waits a short bound for the route that
//! makes its audio to begin, and once it has, for as long as that takes, up to a ceiling: under
//! load speech takes longer than the bound, and the game waits for it. What was handed over is
//! kept for the bound, and the slot of a route that has begun for the ceiling.

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
    Begun,
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
    ceiling: Duration,
    slots: Mutex<HashMap<(Source, String), Slot>>,
}

impl Handoff {
    /// A handoff where a radio route waits up to `bound` for the route that makes its audio to
    /// begin, and up to `ceiling` in all once it has.
    pub fn new(bound: Duration, ceiling: Duration) -> Self {
        Self {
            bound,
            ceiling,
            slots: Mutex::default(),
        }
    }

    /// Says that `source`'s audio for `identifier` is being made, replacing what was handed over
    /// for it before.
    pub fn begin(&self, source: Source, identifier: &str) {
        self.slot(source, identifier, |slot| {
            slot.state.send_replace(State::Begun);
        });
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
        let handed = async {
            let begun = state.wait_for(|state| !matches!(state, State::Pending));
            tokio::time::timeout(self.bound, begun).await.ok()?.ok()?;
            let done = state.wait_for(|state| !matches!(state, State::Pending | State::Begun));
            match &*done.await.ok()? {
                State::Ready(audio) => Some(Ok(audio.clone())),
                State::Failed(status) => Some(Err(Missing::Failed(*status))),
                State::Pending | State::Begun => unreachable!("waited for a state that is done"),
            }
        };
        match tokio::time::timeout(self.ceiling, handed).await {
            Ok(Some(handed)) => handed,
            // A slot is only dropped once it has gone untouched for as long as a wait on it
            // lasts, so only a wait that is timing out anyway can see it closed.
            Ok(None) | Err(_) => Err(Missing::TimedOut),
        }
    }

    /// Calls `f` on the slot of `source` and `identifier`, made if there is none, having dropped
    /// every slot untouched for longer than a wait on it lasts: the ceiling where its route has
    /// begun, else the bound.
    fn slot<T>(&self, source: Source, identifier: &str, f: impl FnOnce(&mut Slot) -> T) -> T {
        let now = Instant::now();
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        slots.retain(|_, slot| {
            let kept = match *slot.state.borrow() {
                State::Begun => self.ceiling,
                _ => self.bound,
            };
            now.duration_since(slot.touched) < kept
        });
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
    const BOUND: Duration = Duration::from_secs(10);
    const CEILING: Duration = Duration::from_secs(30);

    fn audio(n: f32) -> Arc<Audio> {
        Arc::new(Audio {
            rate: dsp::RATE,
            channels: 1,
            samples: vec![n],
        })
    }

    fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        future.poll(&mut Context::from_waker(Waker::noop()))
    }

    #[tokio::test]
    async fn waits_for_what_is_handed_over() {
        let handoff = Handoff::new(LONG, LONG);
        let mut wait = pin!(handoff.wait(Source::Speech, "id"));
        assert!(poll_once(wait.as_mut()).is_pending());
        handoff.publish(Source::Speech, "id", Ok(audio(1.0)));
        assert_eq!(wait.await, Ok(audio(1.0)));
    }

    #[tokio::test]
    async fn gives_what_was_handed_over_before() {
        let handoff = Handoff::new(LONG, LONG);
        handoff.publish(Source::Blips, "id", Ok(audio(1.0)));
        handoff.publish(Source::Speech, "id", Ok(audio(2.0)));
        assert_eq!(handoff.wait(Source::Blips, "id").await, Ok(audio(1.0)));
        assert_eq!(handoff.wait(Source::Speech, "id").await, Ok(audio(2.0)));
        handoff.publish(Source::Speech, "id", Ok(audio(3.0)));
        assert_eq!(handoff.wait(Source::Speech, "id").await, Ok(audio(3.0)));
    }

    #[tokio::test]
    async fn every_waiter_gets_it() {
        let handoff = Handoff::new(LONG, LONG);
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
        let handoff = Handoff::new(LONG, LONG);
        let mut wait = pin!(handoff.wait(Source::Blips, "id"));
        assert!(poll_once(wait.as_mut()).is_pending());
        handoff.publish(Source::Blips, "id", Err(500));
        assert_eq!(wait.await, Err(Missing::Failed(500)));
    }

    #[tokio::test]
    async fn times_out() {
        let handoff = Handoff::new(Duration::from_millis(1), LONG);
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

    #[tokio::test(start_paused = true)]
    async fn times_out_at_the_bound_when_nothing_begins() {
        let handoff = Handoff::new(BOUND, CEILING);
        let started = Instant::now();
        let wait = handoff.wait(Source::Speech, "id");
        let other = async {
            tokio::time::sleep(BOUND / 2).await;
            handoff.begin(Source::Blips, "id");
            handoff.begin(Source::Speech, "other");
        };
        assert_eq!(tokio::join!(wait, other).0, Err(Missing::TimedOut));
        assert_eq!(started.elapsed(), BOUND);
    }

    #[tokio::test(start_paused = true)]
    async fn waits_past_the_bound_for_what_has_begun() {
        for begun_first in [true, false] {
            let handoff = Handoff::new(BOUND, CEILING);
            if begun_first {
                handoff.begin(Source::Speech, "id");
            }
            let wait = handoff.wait(Source::Speech, "id");
            let made = async {
                tokio::time::sleep(BOUND / 2).await;
                handoff.begin(Source::Speech, "id");
                tokio::time::sleep(BOUND * 2).await;
                handoff.publish(Source::Speech, "id", Ok(audio(1.0)));
            };
            assert_eq!(tokio::join!(wait, made).0, Ok(audio(1.0)));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn passes_a_late_failure_on() {
        let handoff = Handoff::new(BOUND, CEILING);
        handoff.begin(Source::Speech, "id");
        let wait = handoff.wait(Source::Speech, "id");
        let failed = async {
            tokio::time::sleep(BOUND * 2).await;
            handoff.publish(Source::Speech, "id", Err(503));
        };
        assert_eq!(tokio::join!(wait, failed).0, Err(Missing::Failed(503)));
    }

    #[tokio::test(start_paused = true)]
    async fn times_out_at_the_ceiling_when_what_has_begun_never_ends() {
        let handoff = Handoff::new(BOUND, CEILING);
        handoff.begin(Source::Speech, "id");
        let started = Instant::now();
        let wait = handoff.wait(Source::Speech, "id");
        assert_eq!(wait.await, Err(Missing::TimedOut));
        assert_eq!(started.elapsed(), CEILING);
    }

    #[tokio::test(start_paused = true)]
    async fn keeps_slots_only_as_long_as_a_wait_on_them() {
        let handoff = Handoff::new(BOUND, CEILING);
        handoff.begin(Source::Speech, "begun");
        tokio::time::sleep(BOUND * 2).await;
        handoff.publish(Source::Speech, "handed", Ok(audio(1.0)));
        assert_eq!(handoff.len(), 2);
        handoff.publish(Source::Speech, "begun", Ok(audio(2.0)));
        tokio::time::sleep(BOUND * 2).await;
        handoff.begin(Source::Speech, "never ends");
        assert_eq!(handoff.len(), 1);
        tokio::time::sleep(CEILING * 2).await;
        handoff.publish(Source::Speech, "last", Ok(audio(3.0)));
        assert_eq!(handoff.len(), 1);
    }
}
