//! Per-request metrics in a SQLite database: what each request asked for, how it went and where
//! its time went, but never what was said.
//!
//! Rows are written by a thread of their own, a batch to a transaction. A request hands its row
//! over without waiting; if the writer has fallen so far behind that its queue is full, the row
//! is dropped with a warning rather than holding the request up.

use std::{
    path::Path,
    sync::mpsc::{self, SyncSender, TrySendError},
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, TransactionBehavior, params};
use tracing::{error, warn};

/// The most rows written in one transaction.
const BATCH: usize = 256;

const PRAGMAS: &str = "
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
";

/// What each version of the schema adds to the one before. A database records the version it is
/// at as its `user_version`, which databases made before there were versions have at 0, with the
/// first version's table.
const MIGRATIONS: [&str; 2] = [
    "CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY,
        -- When the request arrived, in milliseconds since the Unix epoch.
        time_unix_ms INTEGER NOT NULL,
        route TEXT NOT NULL,
        voice TEXT NOT NULL,
        identifier TEXT NOT NULL,
        status INTEGER NOT NULL,
        -- Whether the speech came from the cache, on the route that synthesizes; NULL elsewhere.
        cache_hit INTEGER,
        -- The length of the text in characters.
        text_length INTEGER NOT NULL,
        -- The length of the audio served, NULL where none was.
        audio_seconds REAL,
        -- Waiting for a turn at the model or the CPU.
        queue_ms REAL NOT NULL,
        -- On the radio routes, waiting for the speech or blips to put on the radio.
        handoff_ms REAL NOT NULL,
        synthesis_ms REAL NOT NULL,
        dsp_ms REAL NOT NULL,
        ffmpeg_ms REAL NOT NULL,
        total_ms REAL NOT NULL,
        -- The patches the model generated, NULL where it did not run.
        patches INTEGER,
        -- Bringing the audio to the service's loudness, NULL where it was not: the gain before
        -- the true-peak limiter, the most the limiter took off, and the fraction of the samples
        -- it lowered.
        gain_db REAL,
        limiter_reduction_db REAL,
        limited_fraction REAL
    );",
    "
    -- On the route that synthesizes, how many takes of the voice and text the cache held when
    -- the speech was looked up, 0 to 5; it is given one of them once it holds all 5.
    ALTER TABLE requests ADD COLUMN cache_takes INTEGER;
    -- The RMS level of the audio served, over all of it, in dB from full scale, a full-scale
    -- square wave at 0 and digital silence at -Inf: on the routes that make speech or blips, of
    -- them without the clicks the radio special filter puts around them, and on the radio routes,
    -- of the radio, static and all.
    ALTER TABLE requests ADD COLUMN rms_dbfs REAL;
    ",
];

const INSERT: &str = "
    INSERT INTO requests (
        time_unix_ms, route, voice, identifier, status, cache_hit, text_length, audio_seconds,
        queue_ms, handoff_ms, synthesis_ms, dsp_ms, ffmpeg_ms, total_ms, patches, gain_db,
        limiter_reduction_db, limited_fraction, cache_takes, rms_dbfs
    ) VALUES (
        ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20
    )
";

/// One request's metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// When it arrived.
    pub time: SystemTime,
    /// The route's name, such as `tts-radio`.
    pub route: &'static str,
    pub voice: String,
    pub identifier: String,
    pub status: u16,
    pub cache_hit: Option<bool>,
    /// How many takes the cache held when it was looked up.
    pub cache_takes: Option<usize>,
    /// The length of the text in characters.
    pub text_length: usize,
    pub audio_seconds: Option<f64>,
    /// The RMS level of the audio served, in dBFS.
    pub rms_dbfs: Option<f64>,
    pub timings: Timings,
    pub patches: Option<usize>,
    pub leveling: Option<Leveling>,
}

/// What bringing a request's audio to the service's loudness took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Leveling {
    /// The gain before the limiter, in dB.
    pub gain: f64,
    /// The most the limiter took off, in dB.
    pub reduction: f64,
    /// The fraction of the samples the limiter lowered.
    pub limited: f64,
}

/// Where a request's time went.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    /// Waiting for a turn at the model or the CPU.
    pub queue: Duration,
    /// Waiting for the audio a radio route puts on the radio.
    pub handoff: Duration,
    pub synthesis: Duration,
    pub dsp: Duration,
    pub ffmpeg: Duration,
    pub total: Duration,
}

enum Message {
    Row(Box<Row>),
    Flush(SyncSender<()>),
}

/// The metrics database, and the thread writing to it.
#[derive(Debug)]
pub struct Metrics {
    queue: Option<SyncSender<Message>>,
    writer: Option<JoinHandle<()>>,
}

impl Metrics {
    /// Opens the database at `path`, creating it and its table if need be and bringing it to the
    /// latest schema, with a queue of up to `capacity` rows waiting to be written.
    pub fn open(path: &Path, capacity: usize) -> rusqlite::Result<Self> {
        let mut connection = Connection::open(path)?;
        connection.execute_batch(PRAGMAS)?;
        migrate(&mut connection)?;
        let (queue, rows) = mpsc::sync_channel(capacity);
        let writer = thread::Builder::new()
            .name("metrics".into())
            .spawn(move || write(connection, rows))
            .expect("a thread can be spawned");
        Ok(Self {
            queue: Some(queue),
            writer: Some(writer),
        })
    }

    /// Hands `row` to the writer. Says whether it was taken, which it is not when the queue is
    /// full.
    pub fn record(&self, row: Row) -> bool {
        let queue = self
            .queue
            .as_ref()
            .expect("the queue lives as long as self");
        match queue.try_send(Message::Row(Box::new(row))) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                warn!("metrics: dropped a row, the writer is behind");
                false
            }
            Err(TrySendError::Disconnected(_)) => {
                warn!("metrics: dropped a row, the writer has stopped");
                false
            }
        }
    }

    /// Blocks until every row taken so far has been written, or failed to be.
    pub fn flush(&self) {
        let (done, wait) = mpsc::sync_channel(1);
        let queue = self
            .queue
            .as_ref()
            .expect("the queue lives as long as self");
        if queue.send(Message::Flush(done)).is_ok() {
            let _ = wait.recv();
        }
    }
}

impl Drop for Metrics {
    fn drop(&mut self) {
        // The writer stops once the queue is closed and drained.
        drop(self.queue.take());
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

/// Brings the database to the latest version of the schema. One a later build has brought
/// further is left as it is, since the rows this one writes name their columns.
fn migrate(connection: &mut Connection) -> rusqlite::Result<()> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let version: i64 = transaction.pragma_query_value(None, "user_version", |r| r.get(0))?;
    for (to, migration) in (1..).zip(MIGRATIONS) {
        if version < to {
            transaction.execute_batch(migration)?;
            transaction.pragma_update(None, "user_version", to)?;
        }
    }
    transaction.commit()
}

fn write(mut connection: Connection, messages: mpsc::Receiver<Message>) {
    while let Ok(first) = messages.recv() {
        let mut batch = vec![first];
        while batch.len() < BATCH {
            match messages.try_recv() {
                Ok(message) => batch.push(message),
                Err(_) => break,
            }
        }
        let rows = batch.iter().filter_map(|m| match m {
            Message::Row(row) => Some(&**row),
            Message::Flush(_) => None,
        });
        if let Err(e) = insert(&mut connection, rows) {
            error!("metrics: could not write {} rows: {e}", batch.len());
        }
        for message in batch {
            if let Message::Flush(done) = message {
                let _ = done.send(());
            }
        }
    }
}

fn insert<'a>(
    connection: &mut Connection,
    rows: impl Iterator<Item = &'a Row>,
) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    {
        let mut insert = transaction.prepare(INSERT)?;
        for row in rows {
            let ms = |d: Duration| d.as_secs_f64() * 1000.0;
            let time = row.time.duration_since(UNIX_EPOCH).unwrap_or_default();
            let t = &row.timings;
            let leveling = row.leveling.as_ref();
            insert.execute(params![
                i64::try_from(time.as_millis()).unwrap_or(i64::MAX),
                row.route,
                row.voice,
                row.identifier,
                row.status,
                row.cache_hit,
                row.text_length as i64,
                row.audio_seconds,
                ms(t.queue),
                ms(t.handoff),
                ms(t.synthesis),
                ms(t.dsp),
                ms(t.ffmpeg),
                ms(t.total),
                row.patches.map(|p| p as i64),
                leveling.map(|l| l.gain),
                leveling.map(|l| l.reduction),
                leveling.map(|l| l.limited),
                row.cache_takes.map(|t| t as i64),
                row.rms_dbfs,
            ])?;
        }
    }
    transaction.commit()
}

#[cfg(test)]
mod tests {
    use rusqlite::types::Value;

    use super::*;

    fn row(n: u16) -> Row {
        Row {
            time: UNIX_EPOCH + Duration::from_millis(1_792_950_000_123),
            route: "tts",
            voice: "Man (Ace Detective)".into(),
            identifier: format!("id{n}"),
            status: 200 + n,
            cache_hit: Some(false),
            cache_takes: Some(3),
            text_length: 13,
            audio_seconds: Some(1.25),
            rms_dbfs: Some(-21.5),
            timings: Timings {
                queue: Duration::from_micros(1500),
                handoff: Duration::ZERO,
                synthesis: Duration::from_millis(900),
                dsp: Duration::from_millis(40),
                ffmpeg: Duration::from_millis(60),
                total: Duration::from_millis(1002),
            },
            patches: Some(25),
            leveling: Some(Leveling {
                gain: -12.5,
                reduction: 0.75,
                limited: 0.03125,
            }),
        }
    }

    fn rows(path: &Path) -> Vec<Vec<Value>> {
        let connection = Connection::open(path).unwrap();
        let mut select = connection
            .prepare("SELECT * FROM requests ORDER BY id")
            .unwrap();
        let columns = select.column_count();
        select
            .query_map([], |r| (0..columns).map(|i| r.get(i)).collect())
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    #[test]
    fn writes_each_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.sqlite");
        let metrics = Metrics::open(&path, 16).unwrap();
        assert!(metrics.record(row(0)));
        let mut radio = row(1);
        radio.route = "tts-radio";
        radio.cache_hit = None;
        radio.cache_takes = None;
        radio.audio_seconds = None;
        radio.rms_dbfs = None;
        radio.patches = None;
        radio.leveling = None;
        assert!(metrics.record(radio));
        metrics.flush();
        use Value::{Integer as I, Null, Real as R, Text as T};
        let text = |s: &str| T(s.into());
        assert_eq!(
            rows(&path),
            [
                vec![
                    I(1),
                    I(1_792_950_000_123),
                    text("tts"),
                    text("Man (Ace Detective)"),
                    text("id0"),
                    I(200),
                    I(0),
                    I(13),
                    R(1.25),
                    R(1.5),
                    R(0.0),
                    R(900.0),
                    R(40.0),
                    R(60.0),
                    R(1002.0),
                    I(25),
                    R(-12.5),
                    R(0.75),
                    R(0.03125),
                    I(3),
                    R(-21.5),
                ],
                vec![
                    I(2),
                    I(1_792_950_000_123),
                    text("tts-radio"),
                    text("Man (Ace Detective)"),
                    text("id1"),
                    I(201),
                    Null,
                    I(13),
                    Null,
                    R(1.5),
                    R(0.0),
                    R(900.0),
                    R(40.0),
                    R(60.0),
                    R(1002.0),
                    Null,
                    Null,
                    Null,
                    Null,
                    Null,
                    Null,
                ],
            ]
        );
        drop(metrics);
        // Reopening keeps what was written.
        let metrics = Metrics::open(&path, 16).unwrap();
        metrics.record(row(2));
        drop(metrics);
        assert_eq!(rows(&path).len(), 3);
    }

    /// A database as the service made it before the schema had versions.
    #[test]
    fn keeps_a_database_from_before_versions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.sqlite");
        let old = Connection::open(&path).unwrap();
        old.execute_batch(PRAGMAS).unwrap();
        old.execute_batch(MIGRATIONS[0]).unwrap();
        old.execute_batch(
            "INSERT INTO requests (
                time_unix_ms, route, voice, identifier, status, text_length, queue_ms,
                handoff_ms, synthesis_ms, dsp_ms, ffmpeg_ms, total_ms
            ) VALUES (1, 'tts', 'v', 'old', 200, 5, 0, 0, 0, 0, 0, 0)",
        )
        .unwrap();
        drop(old);
        let metrics = Metrics::open(&path, 16).unwrap();
        let mut silent = row(0);
        silent.rms_dbfs = Some(f64::NEG_INFINITY);
        metrics.record(silent);
        drop(metrics);
        let written = rows(&path);
        assert_eq!(written.len(), 2);
        assert_eq!(written[0][4], Value::Text("old".into()));
        assert_eq!(written[0][19..], [Value::Null, Value::Null]);
        assert_eq!(
            written[1][19..],
            [Value::Integer(3), Value::Real(f64::NEG_INFINITY)]
        );
        let latest = MIGRATIONS.len() as i64;
        let version = |path: &Path| -> i64 {
            let connection = Connection::open(path).unwrap();
            connection
                .pragma_query_value(None, "user_version", |r| r.get(0))
                .unwrap()
        };
        assert_eq!(version(&path), latest);
        // Reopening migrates nothing again.
        drop(Metrics::open(&path, 16).unwrap());
        assert_eq!(rows(&path).len(), 2);
        // A database a later build has brought further keeps its version.
        let later = Connection::open(&path).unwrap();
        later
            .pragma_update(None, "user_version", latest + 1)
            .unwrap();
        drop(later);
        drop(Metrics::open(&path, 16).unwrap());
        assert_eq!(version(&path), latest + 1);
    }

    /// With the database locked, the writer is held in its first transaction with at most a
    /// batch, and the queue fills behind it: of a batch and the queue and ten more, at least ten
    /// are dropped. What was taken is written once the lock goes.
    #[test]
    fn drops_rows_when_the_writer_is_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.sqlite");
        let capacity = 2;
        let metrics = Metrics::open(&path, capacity).unwrap();
        let lock = Connection::open(&path).unwrap();
        lock.execute_batch("BEGIN EXCLUSIVE").unwrap();
        let sent = BATCH + capacity + 10;
        let kept = (0..sent as u16).filter(|&n| metrics.record(row(n))).count();
        assert!(
            (capacity..=sent - 10).contains(&kept),
            "{kept} of {sent} rows taken"
        );
        lock.execute_batch("COMMIT").unwrap();
        metrics.flush();
        assert_eq!(rows(&path).len(), kept);
    }
}
