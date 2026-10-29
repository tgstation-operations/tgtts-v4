//! A step as data: the tensors it touches, the dispatches over them in order, where each
//! tensor lives, and where a barrier must go. Nothing here touches a GPU.
//!
//! A tensor holds elements of one type in a shape, its bytes theirs, and the host names it by a
//! [`Tensor`] of that type. A tensor is not a byte count:
//!
//! ```compile_fail,E0308
//! use graph::graph::{Graph, Kind};
//! let mut g = Graph::default();
//! let x = g.tensor::<vulkan::F16>("x", 4096u64, Kind::Input);
//! ```

use std::{fmt, marker::PhantomData, ops::Range};

use vulkan::{Element, ElementType, Residence, SpecValue};

use crate::kernels::{Access, KernelInfo};

/// A tensor of a [`Graph`], by its index, whatever it holds: as the planner and an op's bindings
/// take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub usize);

/// A tensor of a [`Graph`] that holds elements of `E`: what the host writes and reads it as.
pub struct Tensor<E> {
    id: TensorId,
    element: PhantomData<fn() -> E>,
}

impl<E> Tensor<E> {
    pub fn id(self) -> TensorId {
        self.id
    }
}

impl<E> Clone for Tensor<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E> Copy for Tensor<E> {}

impl<E> PartialEq for Tensor<E> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<E> Eq for Tensor<E> {}

impl<E: Element> fmt::Debug for Tensor<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}>({})", E::TYPE, self.id.0)
    }
}

/// A tensor an op binds: by its id, or by the handle of what it holds.
pub trait Bound {
    fn id(&self) -> TensorId;
}

impl Bound for TensorId {
    fn id(&self) -> TensorId {
        *self
    }
}

impl<E> Bound for Tensor<E> {
    fn id(&self) -> TensorId {
        self.id
    }
}

/// A name for a tensor of elements of `E`: its handle, or its id, which no element type is checked
/// against.
pub trait Holds<E>: Bound {}

impl<E> Holds<E> for Tensor<E> {}

impl<E> Holds<E> for TensorId {}

/// The tensors an op binds, one per binding of its kernel in order: a tuple of them, or a slice
/// or array of one kind.
pub trait Bindings {
    fn ids(&self) -> Vec<TensorId>;
}

impl<B: Bound> Bindings for &[B] {
    fn ids(&self) -> Vec<TensorId> {
        self.iter().map(Bound::id).collect()
    }
}

impl<B: Bound, const N: usize> Bindings for &[B; N] {
    fn ids(&self) -> Vec<TensorId> {
        self.iter().map(Bound::id).collect()
    }
}

macro_rules! tuple_bindings {
    ($($t:ident $v:ident),+) => {
        impl<$($t: Bound),+> Bindings for ($($t,)+) {
            fn ids(&self) -> Vec<TensorId> {
                let ($($v,)+) = self;
                vec![$($v.id()),+]
            }
        }
    };
}

tuple_bindings!(A a, B b);
tuple_bindings!(A a, B b, C c);
tuple_bindings!(A a, B b, C c, D d);
tuple_bindings!(A a, B b, C c, D d, E e);
tuple_bindings!(A a, B b, C c, D d, E e, F f);
tuple_bindings!(A a, B b, C c, D d, E e, F f, G g);
tuple_bindings!(A a, B b, C c, D d, E e, F f, G g, H h);
tuple_bindings!(A a, B b, C c, D d, E e, F f, G g, H h, I i);

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

/// What a graph holds of one of its tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub bytes: u64,
    /// What it holds, and its extents; none for a tensor added by its bytes alone
    /// ([`Graph::untyped`]).
    pub element: Option<(ElementType, Vec<usize>)>,
    pub kind: Kind,
}

impl TensorInfo {
    /// Its extents.
    ///
    /// # Panics
    ///
    /// If it was added by its bytes alone.
    pub fn shape(&self) -> &[usize] {
        match &self.element {
            Some((_, shape)) => shape,
            None => panic!("{} was added by its bytes alone", self.name),
        }
    }

    /// The elements it holds.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
    pub tensors: Vec<TensorInfo>,
    pub ops: Vec<Op>,
}

impl Graph {
    /// Adds a tensor of `shape` elements of `E`, named `name`, to live as `kind` says.
    pub fn tensor<E: Element>(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        kind: Kind,
    ) -> Tensor<E> {
        let elements: usize = shape.iter().product();
        self.tensors.push(TensorInfo {
            name: name.into(),
            bytes: elements as u64 * E::TYPE.bytes(),
            element: Some((E::TYPE, shape.to_vec())),
            kind,
        });
        Tensor {
            id: TensorId(self.tensors.len() - 1),
            element: PhantomData,
        }
    }

    /// Adds a tensor of `bytes` bytes, whose element type no op checks.
    pub fn untyped(&mut self, name: impl Into<String>, bytes: u64, kind: Kind) -> TensorId {
        self.tensors.push(TensorInfo {
            name: name.into(),
            bytes,
            element: None,
            kind,
        });
        TensorId(self.tensors.len() - 1)
    }

    /// Appends a dispatch.
    ///
    /// # Panics
    ///
    /// If the bindings do not match the kernel's in number or in element type, or name a tensor
    /// this graph does not have.
    pub fn op(
        &mut self,
        label: impl Into<String>,
        pipeline: Pipeline,
        bindings: impl Bindings,
        groups: [u32; 3],
    ) {
        let label = label.into();
        let bindings = bindings.ids();
        let kernel = pipeline.kernel;
        assert_eq!(
            bindings.len(),
            kernel.bindings.len(),
            "{label}: {} bindings for {}",
            bindings.len(),
            kernel.name
        );
        for (i, (t, binding)) in bindings.iter().zip(kernel.bindings).enumerate() {
            let tensor = self.tensors.get(t.0);
            let tensor = tensor.unwrap_or_else(|| panic!("{label}: no tensor {}", t.0));
            if let Some((element, _)) = &tensor.element {
                assert!(
                    *element == binding.element,
                    "{label}: {}, bound as binding {i} of {}, holds {element}, not {}",
                    tensor.name,
                    kernel.name,
                    binding.element
                );
            }
        }
        self.ops.push(Op {
            label,
            pipeline,
            bindings,
            groups,
        });
    }

    /// The tensors op `i` reads and those it writes.
    fn accesses(&self, i: usize) -> impl Iterator<Item = (TensorId, Access)> + '_ {
        let op = &self.ops[i];
        let access = op.pipeline.kernel.bindings.iter().map(|b| b.access);
        op.bindings.iter().copied().zip(access)
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
    use vulkan::F16;

    use super::*;
    use crate::kernels::{read, write};

    static COPY: KernelInfo = KernelInfo {
        name: "copy",
        bindings: &[read::<F16>(), write::<F16>()],
    };
    static ADD: KernelInfo = KernelInfo {
        name: "add",
        bindings: &[read::<F16>(), read::<F16>(), write::<F16>()],
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
        let x = g.tensor::<F16>("x", &[500], Kind::Input).id();
        let a = g.tensor::<F16>("a", &[500], Kind::Intermediate).id();
        let b = g.tensor::<F16>("b", &[500], Kind::Intermediate).id();
        let c = g.tensor::<F16>("c", &[500], Kind::Intermediate).id();
        let out = g.tensor::<F16>("out", &[500], Kind::Output).id();
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
        let x = g.tensor::<F16>("x", &[32], Kind::Input).id();
        let y = g.tensor::<F16>("y", &[32], Kind::Intermediate).id();
        let z = g.tensor::<F16>("z", &[32], Kind::Intermediate).id();
        let out = g.tensor::<F16>("out", &[32], Kind::Output).id();
        let w = g.tensor::<F16>("w", &[32], Kind::Constant).id();
        let cache = g.tensor::<F16>("cache", &[32], Kind::State).id();
        let position = g.tensor::<F16>("position", &[2], Kind::Param).id();
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
        let x = p.tensor::<F16>("x", &[500], Kind::Input).id();
        let a = p.tensor::<F16>("a", &[500], Kind::Intermediate).id();
        let b = p.tensor::<F16>("b", &[1000], Kind::Intermediate).id();
        let c = p.tensor::<F16>("c", &[500], Kind::Intermediate).id();
        let out = p.tensor::<F16>("out", &[1000], Kind::Output).id();
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
    #[should_panic(expected = "copy: floats, bound as binding 1 of copy, holds f32, not f16")]
    fn an_op_refuses_a_tensor_of_another_element_type_than_its_kernel_binds() {
        let mut g = Graph::default();
        let x = g.tensor::<F16>("x", &[32], Kind::Input);
        let out = g.tensor::<F16>("out", &[32], Kind::Output);
        let floats = g.tensor::<f32>("floats", &[32], Kind::Output);
        g.op("ok", pipeline(&COPY), (x, out), [1, 1, 1]);
        g.op("copy", pipeline(&COPY), (x, floats), [1, 1, 1]);
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
        let x = g.tensor::<F16>("x", &[32], Kind::Input).id();
        let a = g.tensor::<F16>("a", &[32], Kind::Output).id();
        let b = g.tensor::<F16>("b", &[32], Kind::Output).id();
        let c = g.tensor::<F16>("c", &[32], Kind::Output).id();
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
        let x = g.tensor::<F16>("x", &[128], Kind::Input).id();
        let a = g.tensor::<F16>("a", &[128], Kind::Intermediate).id();
        let b = g.tensor::<F16>("b", &[128], Kind::Output).id();
        let w = g.tensor::<F16>("w", &[128], Kind::Constant).id();
        let c = g.tensor::<F16>("c", &[128], Kind::Intermediate).id();
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
