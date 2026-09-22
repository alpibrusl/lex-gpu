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

use tile_ir::{DType, Space};

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
}

impl Arg {
    pub fn var(self) -> Var {
        match self {
            Arg::Move(v) | Arg::Borrow(v) | Arg::BorrowElem(v, _) => v,
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
    /// Elementwise; `b` may be a row vector `[m]` broadcast over `[m,n]`.
    Binary(BinOp, Arg, Arg),
    Exp(Arg),
    Scale(Arg, f32),
    /// `[m,n] -> [m]`.
    RowReduce(Reduce, Arg),
    /// Explicit dtype conversion, the only way to narrow.
    Convert(Arg, DType),
    /// Pack tiles into an array (all consumed).
    MakeArray(Vec<Var>),
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
        end: usize,
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
}

impl Builder {
    pub fn new(name: &str) -> Builder {
        Builder {
            name: name.to_string(),
            params: vec![],
            names: vec![],
            declared: vec![],
            stack: vec![vec![]],
        }
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
        }
    }
}
