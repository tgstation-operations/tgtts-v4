//! A step as data: the tensors it touches, the dispatches over them in order, where each
//! tensor lives, and where a barrier must go. Nothing here touches a GPU.

use std::ops::Range;

use vulkan::{Residence, SpecValue};

use crate::kernels::{Access, KernelInfo};

/// A tensor of a [`Graph`], by its index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub usize);

/// What a tensor holds, which decides where it lives and for how long.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Written by the host once, read by every step: a weight, a table.
    Constant,
    /// Kept from one step to the next, such as a KV cache.
    State,
    /// Written by the host before each step, such as the position: small, and read by the
    /// dispatches that need it rather than baked into them.
    Param,
    /// Written by the host before each step.
    Input,
    /// Read by the host after each step.
    Output,
    /// Written and read within the step, in the arena.
    Intermediate,
}

impl Kind {
    /// Whether tensors of this kind share the step's one arena, each for its lifetime.
    pub fn in_arena(self) -> bool {
        self == Kind::Intermediate
    }

    /// Where a tensor of this kind lives: the GPU's own memory for what only the GPU touches
    /// every step, mapped memory for what the host writes or reads every step.
    pub fn residence(self) -> Residence {
        match self {
            Kind::Constant | Kind::State | Kind::Intermediate => Residence::Device,
            Kind::Param | Kind::Input => Residence::Upload,
            Kind::Output => Residence::Readback,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub bytes: u64,
    pub kind: Kind,
}

/// A pipeline: a kernel at a choice of specialization constants and subgroup size.
#[derive(Clone, Debug)]
pub struct Pipeline {
    pub kernel: &'static KernelInfo,
    pub constants: Vec<(u32, SpecValue)>,
    pub subgroup_size: Option<u32>,
}

impl Pipeline {
    /// What tells two pipelines apart, for building each once.
    pub fn key(&self) -> (&'static str, Vec<(u32, u32)>, Option<u32>) {
        let bits = |v: SpecValue| match v {
            SpecValue::Bool(b) => u32::from(b),
            SpecValue::U32(v) => v,
            SpecValue::I32(v) => v as u32,
            SpecValue::F32(v) => v.to_bits(),
        };
        let constants = self.constants.iter().map(|&(id, v)| (id, bits(v)));
        (self.kernel.name, constants.collect(), self.subgroup_size)
    }
}

/// One dispatch.
#[derive(Clone, Debug)]
pub struct Op {
    pub label: String,
    pub pipeline: Pipeline,
    /// One tensor per binding of the kernel, in its order.
    pub bindings: Vec<TensorId>,
    pub groups: [u32; 3],
}

/// The dispatches of a step at fixed shapes, in the order they run.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
}

impl Graph {
    pub fn tensor(&mut self, name: impl Into<String>, bytes: u64, kind: Kind) -> TensorId {
        self.tensors.push(Tensor {
            name: name.into(),
            bytes,
            kind,
        });
        TensorId(self.tensors.len() - 1)
    }

    /// Appends a dispatch.
    ///
    /// # Panics
    ///
    /// If the bindings do not match the kernel's, or name a tensor this graph does not have.
    pub fn op(
        &mut self,
        label: impl Into<String>,
        pipeline: Pipeline,
        bindings: &[TensorId],
        groups: [u32; 3],
    ) {
        let label = label.into();
        assert_eq!(
            bindings.len(),
            pipeline.kernel.access.len(),
            "{label}: {} bindings for {}",
            bindings.len(),
            pipeline.kernel.name
        );
        for t in bindings {
            assert!(t.0 < self.tensors.len(), "{label}: no tensor {}", t.0);
        }
        self.ops.push(Op {
            label,
            pipeline,
            bindings: bindings.to_vec(),
            groups,
        });
    }

    /// The tensors op `i` reads and those it writes.
    fn accesses(&self, i: usize) -> impl Iterator<Item = (TensorId, Access)> + '_ {
        let op = &self.ops[i];
        op.bindings
            .iter()
            .copied()
            .zip(op.pipeline.kernel.access.iter().copied())
    }

    /// The ops each arena tensor is live across, first to last inclusive. None for a tensor
    /// outside the arena or no op uses.
    pub fn lifetimes(&self) -> Vec<Option<Range<usize>>> {
        let mut spans: Vec<Option<(usize, usize)>> = vec![None; self.tensors.len()];
        for i in 0..self.ops.len() {
            for (t, _) in self.accesses(i) {
                let span = spans[t.0].get_or_insert((i, i));
                span.1 = i;
            }
        }
        self.tensors
            .iter()
            .zip(spans)
            .map(|(tensor, span)| {
                let (first, end) = span?;
                tensor.kind.in_arena().then_some(first..end + 1)
            })
            .collect()
    }
}

/// Where each arena tensor lives in the arena, and how large it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Each tensor's offset in the arena, None outside it.
    pub offsets: Vec<Option<u64>>,
    pub arena_bytes: u64,
}

/// Offsets in the arena are multiples of this, which every device's storage-buffer offset
/// alignment divides.
pub const ALIGN: u64 = 256;

/// Lays the arena tensors out in one arena, two sharing bytes only if no op sees both live:
/// the largest placed first, each at the lowest aligned offset clear of those already placed
/// whose lifetimes meet its own.
pub fn plan(graph: &Graph) -> Plan {
    plan_programs(&[graph], 0)
}

/// Lays out the arena tensors from `first` on of `graphs`, programs over the same tensors of
/// which one runs at a time, as [`plan`] lays out one graph's: two share bytes only if no
/// program sees both live at once. Those before `first` are left out of the arena.
///
/// # Panics
///
/// If the graphs' tensors differ in number, size or kind.
pub fn plan_programs(graphs: &[&Graph], first: usize) -> Plan {
    let shape =
        |g: &Graph| -> Vec<(u64, Kind)> { g.tensors.iter().map(|t| (t.bytes, t.kind)).collect() };
    let tensors = graphs.first().map_or(0, |g| g.tensors.len());
    assert!(
        graphs.iter().all(|g| shape(g) == shape(graphs[0])),
        "programs over different tensors"
    );
    let lifetimes: Vec<Vec<Option<Range<usize>>>> = graphs.iter().map(|g| g.lifetimes()).collect();
    let live = |t: usize| lifetimes.iter().any(|l| l[t].is_some());
    let meet = |a: usize, b: usize| {
        lifetimes.iter().any(|l| match (&l[a], &l[b]) {
            (Some(a), Some(b)) => a.start < b.end && b.start < a.end,
            _ => false,
        })
    };
    let bytes = |t: usize| graphs[0].tensors[t].bytes;
    let mut order: Vec<usize> = (first..tensors).filter(|&t| live(t)).collect();
    order.sort_by_key(|&t| (std::cmp::Reverse(bytes(t)), t));
    let mut offsets = vec![None; tensors];
    let mut placed: Vec<(usize, Range<u64>)> = Vec::new();
    let mut arena_bytes = 0;
    for t in order {
        let size = bytes(t).max(1).next_multiple_of(ALIGN);
        let mut clashes: Vec<Range<u64>> = placed
            .iter()
            .filter(|(other, _)| meet(t, *other))
            .map(|(_, bytes)| bytes.clone())
            .collect();
        clashes.sort_by_key(|r| r.start);
        let mut offset = 0;
        for r in &clashes {
            if offset + size <= r.start {
                break;
            }
            offset = offset.max(r.end);
        }
        offsets[t] = Some(offset);
        arena_bytes = arena_bytes.max(offset + size);
        placed.push((t, offset..offset + size));
    }
    Plan {
        offsets,
        arena_bytes,
    }
}

/// Where a tensor's bytes are: a range of the arena, or a buffer of its own.
fn region(graph: &Graph, plan: &Plan, t: TensorId) -> (Option<TensorId>, Range<u64>) {
    let bytes = graph.tensors[t.0].bytes;
    match plan.offsets[t.0] {
        Some(offset) => (None, offset..offset + bytes),
        None => (Some(t), 0..bytes),
    }
}

/// Whether each op needs a barrier before it: when it reads what an op since the last barrier
/// wrote, or writes what one since then read or wrote, judged by bytes, since tensors that
/// share the arena share them.
pub fn barriers(graph: &Graph, plan: &Plan) -> Vec<bool> {
    type Region = (Option<TensorId>, Range<u64>);
    let meets = |a: &Region, b: &Region| a.0 == b.0 && a.1.start < b.1.end && b.1.start < a.1.end;
    let mut written: Vec<Region> = Vec::new();
    let mut read: Vec<Region> = Vec::new();
    (0..graph.ops.len())
        .map(|i| {
            let accesses: Vec<(Region, Access)> = graph
                .accesses(i)
                .map(|(t, access)| (region(graph, plan, t), access))
                .collect();
            let hazard = accesses.iter().any(|(r, access)| {
                written.iter().any(|w| meets(r, w))
                    || (*access == Access::Write && read.iter().any(|x| meets(r, x)))
            });
            if hazard {
                written.clear();
                read.clear();
            }
            for (r, access) in accesses {
                match access {
                    Access::Read => read.push(r),
                    Access::Write => written.push(r),
                }
            }
            hazard
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    static COPY: KernelInfo = KernelInfo {
        name: "copy",
        access: &[Access::Read, Access::Write],
    };
    static ADD: KernelInfo = KernelInfo {
        name: "add",
        access: &[Access::Read, Access::Read, Access::Write],
    };

    fn pipeline(kernel: &'static KernelInfo) -> Pipeline {
        Pipeline {
            kernel,
            constants: Vec::new(),
            subgroup_size: None,
        }
    }

    /// x -> a -> b -> c -> out, each link a copy: a and c never live together.
    fn chain() -> (Graph, [TensorId; 5]) {
        let mut g = Graph::default();
        let x = g.tensor("x", 1000, Kind::Input);
        let a = g.tensor("a", 1000, Kind::Intermediate);
        let b = g.tensor("b", 1000, Kind::Intermediate);
        let c = g.tensor("c", 1000, Kind::Intermediate);
        let out = g.tensor("out", 1000, Kind::Output);
        for (from, to) in [(x, a), (a, b), (b, c), (c, out)] {
            g.op("copy", pipeline(&COPY), &[from, to], [1, 1, 1]);
        }
        (g, [x, a, b, c, out])
    }

    fn overlaps(plan: &Plan, g: &Graph, s: TensorId, t: TensorId) -> bool {
        let r = |t: TensorId| {
            let o = plan.offsets[t.0].unwrap();
            o..o + g.tensors[t.0].bytes
        };
        let (a, b) = (r(s), r(t));
        a.start < b.end && b.start < a.end
    }

    #[test]
    fn tensors_never_live_together_share_the_arena_and_others_never_do() {
        let (g, [_, a, b, c, _]) = chain();
        let lifetimes = g.lifetimes();
        assert_eq!(lifetimes[a.0], Some(0..2));
        assert_eq!(lifetimes[b.0], Some(1..3));
        assert_eq!(lifetimes[c.0], Some(2..4));
        let plan = plan(&g);
        for (s, t) in [(a, b), (b, c)] {
            assert!(!overlaps(&plan, &g, s, t), "{s:?} and {t:?}");
        }
        assert!(overlaps(&plan, &g, a, c), "a is dead once c is written");
        // Two live at once at most, each rounded up to the alignment.
        assert_eq!(plan.arena_bytes, 2 * 1024);
        assert!(plan.offsets.iter().flatten().all(|o| o % ALIGN == 0));
    }

    #[test]
    fn what_the_host_reaches_and_what_outlives_a_step_have_buffers_of_their_own() {
        let mut g = Graph::default();
        let x = g.tensor("x", 64, Kind::Input);
        let y = g.tensor("y", 64, Kind::Intermediate);
        let z = g.tensor("z", 64, Kind::Intermediate);
        let out = g.tensor("out", 64, Kind::Output);
        let w = g.tensor("w", 64, Kind::Constant);
        let cache = g.tensor("cache", 64, Kind::State);
        let position = g.tensor("position", 4, Kind::Param);
        g.op("a", pipeline(&ADD), &[w, position, y], [1, 1, 1]);
        g.op("b", pipeline(&ADD), &[x, y, out], [1, 1, 1]);
        g.op("c", pipeline(&ADD), &[w, cache, z], [1, 1, 1]);
        g.op("d", pipeline(&COPY), &[z, cache], [1, 1, 1]);
        let lifetimes = g.lifetimes();
        let plan = plan(&g);
        for t in [x, out, w, cache, position] {
            assert_eq!(lifetimes[t.0], None, "{}", g.tensors[t.0].name);
            assert_eq!(plan.offsets[t.0], None, "{}", g.tensors[t.0].name);
        }
        assert!(overlaps(&plan, &g, y, z));
        // Only the host's own tensors are mapped for it.
        let residence = |t: TensorId| g.tensors[t.0].kind.residence();
        for t in [w, cache, y, z] {
            assert_eq!(residence(t), Residence::Device);
        }
        assert_eq!(residence(x), Residence::Upload);
        assert_eq!(residence(position), Residence::Upload);
        assert_eq!(residence(out), Residence::Readback);
    }

    #[test]
    fn programs_share_bytes_between_tensors_none_of_them_sees_live_together() {
        // P: x -> a -> b -> out, and Q: x -> c -> b -> out, over the same tensors: a meets b in
        // P and c meets b in Q, but a and c never meet. b, the largest, is placed first.
        let mut p = Graph::default();
        let x = p.tensor("x", 1000, Kind::Input);
        let a = p.tensor("a", 1000, Kind::Intermediate);
        let b = p.tensor("b", 2000, Kind::Intermediate);
        let c = p.tensor("c", 1000, Kind::Intermediate);
        let out = p.tensor("out", 2000, Kind::Output);
        let mut q = p.clone();
        for (from, to) in [(x, a), (a, b), (b, out)] {
            p.op("copy", pipeline(&COPY), &[from, to], [1, 1, 1]);
        }
        for (from, to) in [(x, c), (c, b), (b, out)] {
            q.op("copy", pipeline(&COPY), &[from, to], [1, 1, 1]);
        }
        let both = plan_programs(&[&p, &q], 0);
        assert!(!overlaps(&both, &p, a, b), "a and b meet in P");
        assert!(!overlaps(&both, &p, b, c), "b and c meet in Q");
        assert!(overlaps(&both, &p, a, c));
        assert_eq!(both.arena_bytes, 3 * 1024);
        // Tensors before `first` stay out of the arena.
        let later = plan_programs(&[&p, &q], c.0);
        assert_eq!(later.offsets[a.0], None);
        assert_eq!(later.offsets[c.0], Some(0));
    }

    #[test]
    fn a_barrier_goes_before_each_op_that_reads_or_overwrites_what_came_before() {
        let (g, _) = chain();
        let plan = plan(&g);
        assert_eq!(barriers(&g, &plan), [false, true, true, true]);
    }

    #[test]
    fn independent_ops_share_no_barrier() {
        let mut g = Graph::default();
        let x = g.tensor("x", 64, Kind::Input);
        let a = g.tensor("a", 64, Kind::Output);
        let b = g.tensor("b", 64, Kind::Output);
        let c = g.tensor("c", 64, Kind::Output);
        g.op("a", pipeline(&COPY), &[x, a], [1, 1, 1]);
        g.op("b", pipeline(&COPY), &[x, b], [1, 1, 1]);
        g.op("c", pipeline(&ADD), &[a, b, c], [1, 1, 1]);
        let plan = plan(&g);
        assert_eq!(barriers(&g, &plan), [false, false, true]);
    }

    #[test]
    fn reusing_bytes_another_op_just_read_needs_a_barrier() {
        // a is read by the second op; c, written by the third, may take a's bytes only once
        // that read is done.
        let mut g = Graph::default();
        let x = g.tensor("x", 256, Kind::Input);
        let a = g.tensor("a", 256, Kind::Intermediate);
        let b = g.tensor("b", 256, Kind::Output);
        let w = g.tensor("w", 256, Kind::Constant);
        let c = g.tensor("c", 256, Kind::Intermediate);
        g.op("a", pipeline(&COPY), &[x, a], [1, 1, 1]);
        g.op("b", pipeline(&COPY), &[a, b], [1, 1, 1]);
        g.op("c", pipeline(&COPY), &[w, c], [1, 1, 1]);
        let plan = plan(&g);
        assert!(overlaps(&plan, &g, a, c));
        assert_eq!(barriers(&g, &plan), [false, true, true]);
        // Planted defect: a planner that put c apart from everything would need no barrier
        // there, so the check is of the bytes, not the names.
        let mut apart = plan.clone();
        apart.offsets[c.0] = Some(plan.arena_bytes);
        assert_eq!(barriers(&g, &apart), [false, true, false]);
    }
}
