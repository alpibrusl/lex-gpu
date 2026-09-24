//! The typed tile program: values, ops, structured regions, and a builder.
//!
//! This is the representation the linearity checker and the interpreter both
//! walk. There is no parser yet — the design doc's v1 frontend is an embedded
//! Rust DSL, and [`Builder`] is the smallest honest version of one.
//!
//! Ownership is explicit at every use site: an operand is either moved
//! ([`Arg::Move`]) or borrowed for the duration of the op ([`Arg::Borrow`],
//! [`Arg::BorrowElem`]). Nothing is implicitly copied; [`Op::Dup`] is the only
//! way to get a second tile with the same contents, and it is counted.

use lex_ir::{DType, Space};

/// An SSA value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var(pub u32);

/// The static type of a tile: element type, shape and memory space.
///
/// Layout is not here yet. Every tile is row-major; the layout algebra lands
/// when a second layout exists for something to be checked against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileTy {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub space: Space,
}

impl TileTy {
    pub fn new(dtype: DType, shape: &[usize], space: Space) -> TileTy {
        TileTy {
            dtype,
            shape: shape.to_vec(),
            space,
        }
    }

    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> usize {
        self.elems() * self.dtype.size_bytes()
    }
}

/// A pipe: a ring of `stages` threadgroup slots, each holding one tile per
/// entry of `elem`, passed from one producer role to `consumers` consumer
/// roles. `id` names the pipe (it is the producer handle's variable).
///
/// Lowered to Hopper, every slot has a *full* barrier (1 arrival plus the
/// copy's transaction bytes) and an *empty* barrier (`consumers` arrivals).
/// Both counts are read off this type; nobody writes them down.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipeTy {
    pub id: Var,
    pub elem: Vec<TileTy>,
    pub stages: usize,
    pub consumers: usize,
}

impl PipeTy {
    pub fn slot_bytes(&self) -> usize {
        self.elem.iter().map(TileTy::bytes).sum()
    }
}

/// The type of an SSA value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    /// A tile. Linear: consumed exactly once.
    Tile(TileTy),
    /// A tile whose contents are still in flight. Linear; the only way to get
    /// at the tile is [`Op::Wait`], which discharges the effect.
    Future(TileTy),
    /// A fixed-length array of linear tiles of one type. Linear as a whole.
    Array(TileTy, usize),
    /// A loop induction variable. Unrestricted; usable only in index
    /// expressions.
    Index,
    /// The producer end of a pipe. Owned by exactly one role.
    Producer(PipeTy),
    /// Consumer end `k` of a pipe. Each is owned by exactly one role, which is
    /// what makes the empty barrier's arrival count right by construction.
    Consumer(PipeTy, usize),
    /// An empty slot the producer has acquired: writable only by `commit`.
    Slot(PipeTy),
    /// Consumer `k`'s read-only share of a filled slot. Must be released.
    Share(PipeTy, usize),
}

impl Ty {
    pub fn is_linear(&self) -> bool {
        !matches!(self, Ty::Index)
    }

    /// Threadgroup bytes this value pins while it is live. A future pins its
    /// destination buffer: the copy is writing into it.
    pub fn threadgroup_bytes(&self) -> usize {
        match self {
            Ty::Tile(t) | Ty::Future(t) if t.space == Space::Threadgroup => t.bytes(),
            Ty::Array(t, n) if t.space == Space::Threadgroup => t.bytes() * n,
            // Slots and shares live in the pipe's ring, which is budgeted
            // once, when the pipe is created.
            _ => 0,
        }
    }
}

/// A global tensor argument of the kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub writable: bool,
}

/// `constant + Σ coeff·var`, over loop induction variables.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct IdxExpr {
    pub constant: i64,
    pub terms: Vec<(Var, i64)>,
}

impl IdxExpr {
    pub fn lit(c: usize) -> IdxExpr {
        IdxExpr {
            constant: c as i64,
            terms: vec![],
        }
    }

    /// `coeff * v + constant`
    pub fn scaled(v: Var, coeff: usize, constant: usize) -> IdxExpr {
        IdxExpr {
            constant: constant as i64,
            terms: vec![(v, coeff as i64)],
        }
    }

    /// `self + coeff * v`
    pub fn plus(mut self, v: Var, coeff: usize) -> IdxExpr {
        self.terms.push((v, coeff as i64));
        self
    }

    /// `self + c`
    pub fn shift(mut self, c: usize) -> IdxExpr {
        self.constant += c as i64;
        self
    }
}

/// A rectangular window into a global tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    pub param: usize,
    pub offset: Vec<IdxExpr>,
    pub shape: Vec<usize>,
}

/// How an op uses one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    /// Consume the value. It may not be used again.
    Move(Var),
    /// Read the value for the duration of the op. It stays live.
    Borrow(Var),
    /// Read element `index` of an array for the duration of the op.
    BorrowElem(Var, Var),
    /// Read tile `part` of a pipe share for the duration of the op.
    BorrowPart(Var, usize),
}

impl Arg {
    pub fn var(self) -> Var {
        match self {
            Arg::Move(v) | Arg::Borrow(v) | Arg::BorrowElem(v, _) | Arg::BorrowPart(v, _) => v,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
}

impl BinOp {
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Max => a.max(b),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::Div => "div",
            BinOp::Max => "max",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Rsqrt,
    Sigmoid,
    /// `log(1 + exp(x))`, evaluated as `max(x, 0) + log(1 + exp(-|x|))` so
    /// a large `x` neither overflows nor loses the linear part. Qwen3.8's
    /// decay gate is `exp(-A softplus(a + bias))`, and its bias reaches 19.
    Softplus,
}

impl UnOp {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            UnOp::Rsqrt => 1.0 / x.sqrt(),
            UnOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnOp::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            UnOp::Rsqrt => "rsqrt",
            UnOp::Sigmoid => "sigmoid",
            UnOp::Softplus => "softplus",
        }
    }
}

/// Which values a packed 4-bit [`Op::Dequant4`] produces from `q: [r, c]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nibbles {
    /// The low nibble of every byte: output `[r, c]`.
    Low,
    /// The high nibble of every byte: output `[r, c]`.
    High,
    /// Both, in order — byte `k` holds columns `2k` (low) and `2k + 1`
    /// (high): output `[r, 2c]`. The layout of the file's own packing, and
    /// with lazy operands the cheapest to read: one byte feeds two values.
    Pairs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduce {
    Max,
    Sum,
}

/// Tile operations. Every op produces zero or one value.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// An uninitialised buffer. Reading it before writing is a runtime error
    /// in the interpreter; making that a type error is a typestate question
    /// left for later.
    Alloc(TileTy),
    Fill(TileTy, f32),
    /// Synchronous global -> tile load.
    Load(View, TileTy),
    /// Start a global -> tile copy into `dst`, which is consumed. The result
    /// is a future holding the same buffer.
    CopyAsync(View, Var),
    /// Discharge a future's effect, returning its tile.
    Wait(Var),
    /// Tile -> global store. Only into a writable parameter, and only at the
    /// parameter's own dtype: narrowing must be an explicit [`Op::Convert`].
    Store(Arg, View),
    /// Consume a value, releasing its storage.
    Drop(Var),
    /// A fresh tile with the same contents. The only way to get two.
    Dup(Arg),
    /// `[m,k] x [n,k]^T -> [m,n]`, accumulating in `acc`.
    MatMulNT(Arg, Arg, DType),
    /// `[m,k] x [k,n] -> [m,n]`, accumulating in `acc`.
    MatMul(Arg, Arg, DType),
    /// Elementwise. `b` may also be `[m]`, one value per row of an `[m,n]`
    /// `a`, or `[1,n]`, one row repeated for every row of `a`.
    Binary(BinOp, Arg, Arg),
    Exp(Arg),
    Unary(UnOp, Arg),
    /// Swap adjacent elements: `out[2i] = a[2i+1]`, `out[2i+1] = a[2i]`.
    /// With a sign-folded sine table this is RoPE's rotation of pairs.
    SwapPairs(Arg),
    /// `q[r,c] * s[r, c / group] - m[r, c / group]` in f32: quantised
    /// values meet their scales (and, for affine formats such as Q4_K, their
    /// mins). The only way to compute with an `I8` tile.
    Dequant(Arg, Arg, Option<Arg>, usize),
    /// [`Op::Dequant`] for unsigned 4-bit values packed two per byte.
    /// Which nibbles it takes is the [`Nibbles`] mode.
    Dequant4(Arg, Arg, Option<Arg>, usize, Nibbles),
    /// NVFP4: 4-bit float values (E2M1: sign, 2-bit exponent, 1-bit
    /// mantissa, magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6) packed two per byte
    /// as [`Nibbles::Pairs`], an FP8 E4M3 scale per `group` values held in
    /// an `I8` tile, and one f32 scale per row (a whole tensor's, repeated).
    /// Value `e2m1(q) * e4m3(s) * gs[row]`; output `[r, c]` in f32.
    /// `q: [r, c/2]`, `s: [r, c/group]`, `gs: [r]`.
    DequantFp4(Arg, Arg, Arg, usize),
    /// Unsigned 6-bit values split into bit planes, as Q6_K stores them: a
    /// 4-bit plane `lo: [r, c/2]` (column `2k` low nibble of byte `k`,
    /// `2k + 1` high) and a 2-bit plane `hi: [r, c/4]` (columns `4k..4k+3`
    /// in bits 0-1, 2-3, 4-5, 6-7 of byte `k`). Value `(lo | hi << 4) - 32`
    /// times `s[r, c / group]`; output `[r, c]` in f32.
    Dequant6(Arg, Arg, Arg, usize),
    Scale(Arg, f32),
    /// `[m,n] -> [m]`.
    RowReduce(Reduce, Arg),
    /// `a[i, j]`, or `-inf` where `first + j >= limit`: masks the columns of
    /// a score block that lie past a runtime length (decode) or past a
    /// query's own position (causal prefill). Both bounds are affine in
    /// loop indices and runtime scalars.
    MaskCols(Arg, IdxExpr, IdxExpr),
    /// Explicit dtype conversion, the only way to narrow.
    Convert(Arg, DType),
    /// Pack tiles into an array (all consumed).
    MakeArray(Vec<Var>),
    /// Producer: wait until a slot is empty, and take it.
    Acquire(Var),
    /// Producer: copy one view per pipe element into the slot (consumed) and
    /// arrive on its full barrier.
    Commit(Var, Vec<View>, Var),
    /// Consumer: wait until the next slot is full, and share it.
    Receive(Var),
    /// Consumer: give the share (consumed) back; arrive on the empty barrier.
    Release(Var, Var),
}

/// A structured statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Let {
        dst: Option<Var>,
        op: Op,
    },
    /// `for index in start..end` with loop-carried linear state.
    ///
    /// `init` is moved into `params` on the first iteration; `body.yields`
    /// are moved into `params` on the next one and into `results` after the
    /// last. The carry type is therefore a loop invariant, and it is checked.
    For {
        index: Var,
        start: usize,
        /// Static upper bound: the checker proves every iteration up to it.
        end: usize,
        /// Runtime end, a dynamic scalar `<= end`. The loop stops at
        /// whichever comes first.
        end_dyn: Option<Var>,
        init: Vec<Var>,
        params: Vec<Var>,
        body: Block,
        results: Vec<Var>,
    },
    /// Run `body` once per element of equal-length arrays, moving element `i`
    /// of each array into `params`. The body yields one tile per output
    /// array; zero yields consumes the arrays outright.
    ///
    /// This is how per-query-block state lives across the KV loop without an
    /// indexed take/put that the checker could not prove disjoint.
    MapEach {
        index: Var,
        arrays: Vec<Var>,
        params: Vec<Var>,
        body: Block,
        results: Vec<Var>,
    },
    /// Split the threadgroup into concurrently running roles connected by
    /// pipes. Handles for every pipe are created here and moved into roles
    /// through `inputs`; each role joins by yielding its `results`.
    Specialize {
        pipes: Vec<PipeDecl>,
        roles: Vec<Role>,
    },
}

/// The handles a pipe hands out: one producer, one per consumer.
#[derive(Clone, Debug, PartialEq)]
pub struct PipeDecl {
    pub ty: PipeTy,
    pub consumers: Vec<Var>,
}

/// One role of a warp-specialised region.
#[derive(Clone, Debug, PartialEq)]
pub struct Role {
    pub name: String,
    /// Simdgroups / warps this role occupies.
    pub warps: usize,
    /// Values moved in (pipe handles included), bound to `params`.
    pub inputs: Vec<Var>,
    pub params: Vec<Var>,
    pub body: Block,
    pub results: Vec<Var>,
}

/// A role under construction: see [`Builder::specialize`].
pub struct RoleDef<'a> {
    pub name: &'a str,
    pub warps: usize,
    /// Values to move in, with their types.
    pub inputs: Vec<(Var, Ty)>,
    pub n_results: usize,
    #[allow(clippy::type_complexity)]
    pub body: Box<dyn FnOnce(&mut Builder, &[Var]) -> Vec<Var> + 'a>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub yields: Vec<Var>,
}

/// A kernel: parameters, a body, and the declared type of every value.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub name: String,
    pub params: Vec<Param>,
    pub body: Block,
    /// Names for printing and diagnostics, indexed by `Var.0`.
    pub names: Vec<String>,
    /// Declared types of region parameters and loop indices, indexed by
    /// `Var.0`. Values defined by ops have `None`: their types are inferred.
    pub declared: Vec<Option<Ty>>,
    /// Independent instances of the body, one threadgroup each on a GPU.
    /// `pid` is the instance index, usable in view offsets like a loop index.
    pub grid: usize,
    pub pid: Option<Var>,
    /// A second, independent grid dimension (`pid2` over `0..grid2`): e.g.
    /// KV heads × cache splits. 1 when unused.
    pub grid2: usize,
    pub pid2: Option<Var>,
    /// Runtime scalars, in buffer order, with their inclusive upper bounds.
    /// Usable like loop indices; the bound is what the checker proves
    /// accesses against.
    pub dyn_scalars: Vec<(Var, usize)>,
}

impl Program {
    pub fn name_of(&self, v: Var) -> &str {
        &self.names[v.0 as usize]
    }
}

/// Builds a [`Program`]. Region bodies are closures, so a Rust function can
/// emit the same block twice — which is how pipeline prologues and epilogues
/// are peeled without the IR needing an `Option<Future>`.
pub struct Builder {
    name: String,
    params: Vec<Param>,
    names: Vec<String>,
    declared: Vec<Option<Ty>>,
    stack: Vec<Vec<Stmt>>,
    grid: usize,
    pid: Option<Var>,
    grid2: usize,
    pid2: Option<Var>,
    dyn_scalars: Vec<(Var, usize)>,
}

impl Builder {
    pub fn new(name: &str) -> Builder {
        Builder {
            name: name.to_string(),
            params: vec![],
            names: vec![],
            declared: vec![],
            stack: vec![vec![]],
            grid: 1,
            pid: None,
            grid2: 1,
            pid2: None,
            dyn_scalars: vec![],
        }
    }

    /// Run the program as `n` independent instances. Returns the instance
    /// index, which views may use like a loop index over `0..n`.
    pub fn grid(&mut self, n: usize) -> Var {
        assert!(self.pid.is_none(), "grid declared twice");
        let pid = self.fresh("pid", Some(Ty::Index));
        self.grid = n;
        self.pid = Some(pid);
        pid
    }

    /// A second grid dimension of `n` instances; returns its index. A
    /// runtime may launch fewer than `n` along it (only the splits a
    /// sequence needs, say): every instance is independent.
    pub fn grid2(&mut self, n: usize) -> Var {
        assert!(self.pid2.is_none(), "grid2 declared twice");
        let pid2 = self.fresh("pid2", Some(Ty::Index));
        self.grid2 = n;
        self.pid2 = Some(pid2);
        pid2
    }

    pub fn param(&mut self, name: &str, dtype: DType, shape: &[usize], writable: bool) -> usize {
        self.params.push(Param {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            writable,
        });
        self.params.len() - 1
    }

    fn fresh(&mut self, name: &str, declared: Option<Ty>) -> Var {
        let v = Var(self.names.len() as u32);
        // Suffix with the id so names stay unique when a builder function is
        // called more than once (peeled iterations, per-block loads).
        self.names.push(format!("{name}.{}", v.0));
        self.declared.push(declared);
        v
    }

    fn push(&mut self, s: Stmt) {
        self.stack.last_mut().expect("builder stack").push(s);
    }

    /// Emit an op that produces a value.
    pub fn op(&mut self, name: &str, op: Op) -> Var {
        let v = self.fresh(name, None);
        self.push(Stmt::Let { dst: Some(v), op });
        v
    }

    /// Emit an op for its effect only.
    pub fn effect(&mut self, op: Op) {
        self.push(Stmt::Let { dst: None, op });
    }

    pub fn drop(&mut self, v: Var) {
        self.effect(Op::Drop(v));
    }

    /// A runtime scalar with an inclusive upper bound, read from the
    /// kernel's scalar buffer in declaration order.
    pub fn dyn_index(&mut self, name: &str, max: usize) -> Var {
        let v = self.fresh(name, Some(Ty::Index));
        self.dyn_scalars.push((v, max));
        v
    }

    /// `for i in start..end`, stopping early at the runtime scalar `dyn_end`.
    pub fn for_range_dyn(
        &mut self,
        start: usize,
        end: usize,
        dyn_end: Var,
        init: Vec<Var>,
        carry: Vec<Ty>,
        body: impl FnOnce(&mut Builder, Var, &[Var]) -> Vec<Var>,
    ) -> Vec<Var> {
        let out = self.for_range(start, end, init, carry, body);
        if let Some(Stmt::For { end_dyn, .. }) = self.stack.last_mut().and_then(|s| s.last_mut()) {
            *end_dyn = Some(dyn_end);
        }
        out
    }

    /// `for i in start..end`, carrying `init` (moved) with per-iteration types
    /// `carry`. The closure receives the index and the carried values and
    /// returns the values to carry on.
    pub fn for_range(
        &mut self,
        start: usize,
        end: usize,
        init: Vec<Var>,
        carry: Vec<Ty>,
        body: impl FnOnce(&mut Builder, Var, &[Var]) -> Vec<Var>,
    ) -> Vec<Var> {
        let index = self.fresh("i", Some(Ty::Index));
        let params: Vec<Var> = carry
            .iter()
            .map(|t| self.fresh("carry", Some(t.clone())))
            .collect();
        self.stack.push(vec![]);
        let yields = body(self, index, &params);
        let stmts = self.stack.pop().expect("builder stack");
        let results: Vec<Var> = (0..params.len()).map(|_| self.fresh("out", None)).collect();
        self.push(Stmt::For {
            index,
            start,
            end,
            end_dyn: None,
            init,
            params,
            body: Block { stmts, yields },
            results: results.clone(),
        });
        results
    }

    /// Run `body` over each element of `arrays` (moved). `elem` gives the
    /// element type of each array.
    pub fn map_each(
        &mut self,
        arrays: Vec<Var>,
        elem: Vec<Ty>,
        n_results: usize,
        body: impl FnOnce(&mut Builder, Var, &[Var]) -> Vec<Var>,
    ) -> Vec<Var> {
        let index = self.fresh("qb", Some(Ty::Index));
        let params: Vec<Var> = elem
            .iter()
            .map(|t| self.fresh("elem", Some(t.clone())))
            .collect();
        self.stack.push(vec![]);
        let yields = body(self, index, &params);
        let stmts = self.stack.pop().expect("builder stack");
        let results: Vec<Var> = (0..n_results).map(|_| self.fresh("arr", None)).collect();
        self.push(Stmt::MapEach {
            index,
            arrays,
            params,
            body: Block { stmts, yields },
            results: results.clone(),
        });
        results
    }

    /// Declare a pipe. Its handles are bound by the next
    /// [`Builder::specialize`], which must move each into exactly one role.
    pub fn pipe(&mut self, elem: Vec<TileTy>, stages: usize, consumers: usize) -> PipeDecl {
        let id = self.fresh("prod", None);
        let ty = PipeTy {
            id,
            elem,
            stages,
            consumers,
        };
        self.declared[id.0 as usize] = Some(Ty::Producer(ty.clone()));
        let consumers = (0..consumers)
            .map(|k| self.fresh("cons", Some(Ty::Consumer(ty.clone(), k))))
            .collect();
        PipeDecl { ty, consumers }
    }

    /// Run `roles` concurrently over `pipes`. Returns each role's results.
    pub fn specialize(&mut self, pipes: Vec<PipeDecl>, roles: Vec<RoleDef<'_>>) -> Vec<Vec<Var>> {
        let mut built = vec![];
        let mut outs = vec![];
        for r in roles {
            let (inputs, tys): (Vec<Var>, Vec<Ty>) = r.inputs.into_iter().unzip();
            let params: Vec<Var> = tys
                .into_iter()
                .map(|t| self.fresh(r.name, Some(t)))
                .collect();
            self.stack.push(vec![]);
            let yields = (r.body)(self, &params);
            let stmts = self.stack.pop().expect("builder stack");
            let results: Vec<Var> = (0..r.n_results)
                .map(|_| self.fresh("joined", None))
                .collect();
            outs.push(results.clone());
            built.push(Role {
                name: r.name.to_string(),
                warps: r.warps,
                inputs,
                params,
                body: Block { stmts, yields },
                results,
            });
        }
        self.push(Stmt::Specialize {
            pipes,
            roles: built,
        });
        outs
    }

    pub fn finish(mut self) -> Program {
        let stmts = self.stack.pop().expect("builder stack");
        assert!(self.stack.is_empty(), "unbalanced builder regions");
        Program {
            name: self.name,
            params: self.params,
            body: Block {
                stmts,
                yields: vec![],
            },
            names: self.names,
            declared: self.declared,
            grid: self.grid,
            pid: self.pid,
            grid2: self.grid2,
            pid2: self.pid2,
            dyn_scalars: self.dyn_scalars,
        }
    }
}
