//! The CPU's own counts of what the process did, which other work on the machine barely moves:
//! user-space events of every thread, as perf_event_open(2) gives them to an unprivileged
//! process. They see nothing of work on a GPU.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Read},
    os::fd::FromRawFd,
};

/// The kernel's generic hardware events this records, by perf's names. On Zen the cache
/// events count at L2, which is as far out as a process may look here: the L3 and data-fabric
/// PMUs, where DRAM traffic shows, need system-wide counting.
const EVENTS: [(&str, u64); 4] = [
    ("cycles", 0),
    ("instructions", 1),
    ("cache-references", 2),
    ("cache-misses", 3),
];

const PERF_TYPE_HARDWARE: u32 = 0;
const INHERIT: u64 = 1 << 1;
const EXCLUDE_KERNEL: u64 = 1 << 5;
const EXCLUDE_HV: u64 = 1 << 6;
const FORMAT_TOTAL_TIME_ENABLED: u64 = 1 << 0;
const FORMAT_TOTAL_TIME_RUNNING: u64 = 1 << 1;
const PERF_FLAG_FD_CLOEXEC: libc::c_ulong = 1 << 3;

/// `struct perf_event_attr` at PERF_ATTR_SIZE_VER8, the bitfield as one word.
#[repr(C)]
#[derive(Default)]
struct Attr {
    kind: u32,
    size: u32,
    config: u64,
    sample_period: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup_events: u32,
    bp_type: u32,
    config1: u64,
    config2: u64,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
    sample_regs_intr: u64,
    aux_watermark: u32,
    sample_max_stack: u16,
    reserved_2: u16,
    aux_sample_size: u32,
    reserved_3: u32,
    sig_data: u64,
    config3: u64,
}

pub struct Counters {
    events: Vec<(&'static str, File)>,
}

/// Each event's count and the nanoseconds it was enabled and counting, which differ when the
/// kernel had to share counters among more events than the PMU has.
#[derive(Clone, Debug)]
pub struct Reading(Vec<(&'static str, [u64; 3])>);

impl Counters {
    /// Opens every event the PMU and the kernel's perf_event_paranoid allow, counting the
    /// calling thread and every thread it starts afterwards, so it has to be called before a
    /// thread pool spawns. Gives why each other event was refused.
    pub fn open() -> (Self, BTreeMap<String, String>) {
        let mut events = Vec::new();
        let mut refused = BTreeMap::new();
        for (name, config) in EVENTS {
            match open(config) {
                Ok(file) => events.push((name, file)),
                Err(e) => {
                    refused.insert(name.to_owned(), e.to_string());
                }
            }
        }
        (Self { events }, refused)
    }

    pub fn names(&self) -> Vec<String> {
        self.events
            .iter()
            .map(|(name, _)| (*name).to_owned())
            .collect()
    }

    pub fn read(&self) -> Reading {
        Reading(
            self.events
                .iter()
                .filter_map(|(name, file)| {
                    let mut bytes = [0u8; 24];
                    (&*file).read_exact(&mut bytes).ok()?;
                    let word = |i: usize| {
                        u64::from_ne_bytes(bytes[8 * i..8 * i + 8].try_into().expect("8 bytes"))
                    };
                    Some((*name, [word(0), word(1), word(2)]))
                })
                .collect(),
        )
    }
}

impl Reading {
    /// Each event's count from `self` to `later`, per one of `reps` repetitions, scaled up for
    /// the time it was not counting; leaves out an event that never counted.
    pub fn per_rep(&self, later: &Reading, reps: usize) -> BTreeMap<String, f64> {
        let mut counts = BTreeMap::new();
        for ((name, before), (_, after)) in self.0.iter().zip(&later.0) {
            let [count, enabled, running] = [0, 1, 2].map(|i| after[i].saturating_sub(before[i]));
            if running == 0 {
                continue;
            }
            let scaled = count as f64 * enabled as f64 / running as f64;
            counts.insert((*name).to_owned(), scaled / reps as f64);
        }
        counts
    }
}

fn open(config: u64) -> io::Result<File> {
    let attr = Attr {
        kind: PERF_TYPE_HARDWARE,
        size: size_of::<Attr>() as u32,
        config,
        read_format: FORMAT_TOTAL_TIME_ENABLED | FORMAT_TOTAL_TIME_RUNNING,
        flags: INHERIT | EXCLUDE_KERNEL | EXCLUDE_HV,
        ..Attr::default()
    };
    // SAFETY: attr is a valid perf_event_attr of the size it declares, for the calling thread
    // (pid 0) on any CPU (-1), in no group (-1).
    let fd = unsafe {
        libc::syscall(
            libc::SYS_perf_event_open,
            &attr as *const Attr,
            0,
            -1,
            -1,
            PERF_FLAG_FD_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the kernel just returned this descriptor, which nothing else owns.
    Ok(unsafe { File::from_raw_fd(fd as i32) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attr_is_the_kernels_ver8_size() {
        assert_eq!(size_of::<Attr>(), 136);
    }

    #[test]
    fn counts_are_per_repetition_and_scaled_for_sharing() {
        let before = Reading(vec![("instructions", [100, 1_000, 1_000])]);
        let after = Reading(vec![("instructions", [700, 3_000, 2_000])]);
        let counts = before.per_rep(&after, 2);
        assert_eq!(counts["instructions"], 600.0 * 2.0 / 2.0);
        let idle = Reading(vec![("instructions", [700, 3_000, 1_000])]);
        assert!(before.per_rep(&idle, 1).is_empty());
    }
}
