//! The type checker: linearity, effects, shapes, bounds and budget.
//!
//! Rules, all of them static:
//!
//! - **Linearity.** Every tile, future and array is consumed exactly once —
//!   moved into an op, yielded, or dropped. A second use is a use-after-move;
//!   no use at all is a leak.
//! - **Region-scoped borrows.** A loop body may borrow anything in scope, but
//!   may only *consume* values it defined itself or received as a carried
//!   parameter. Consuming an outer value would consume it once per iteration.
//! - **Effects.** A copy produces a `Future`, which no compute op accepts.
//!   `wait` is the only way to the tile, so a read-before-arrival cannot be
//!   written down.
//! - **Loop-invariant carries.** What a loop body yields must have exactly the
//!   type it was given. This is what makes a rotated double buffer check.
//! - **Numerics.** Dtypes never change implicitly; narrowing needs `convert`.
//! - **Bounds.** View offsets are affine in loop indices with known ranges, so
//!   every view is proven in bounds for every iteration.
//! - **Budget.** Peak live threadgroup bytes, futures' in-flight buffers
//!   included, must fit the target's table.

use std::collections::{HashMap, HashSet};
use std::fmt;

use tile_ir::{DType, Space, Target};

use crate::ir::{Arg, Block, IdxExpr, Op, Program, Stmt, TileTy, Ty, Var, View};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    UseAfterMove,
    Leak,
    /// A loop body consumed a value defined outside it.
    ConsumeOuter,
    /// The same value was moved and borrowed by one op.
    MovedWhileBorrowed,
    Scope,
    Type,
    /// Loop carry changed type across an iteration.
    Carry,
    Shape,
    Narrowing,
    Bounds,
    Budget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diag {
    pub kind: Kind,
    pub msg: String,
}

impl fmt::Display for Diag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.msg)
    }
}

/// What a successful check learned about the program.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub peak_threadgroup_bytes: usize,
    pub moves: usize,
    pub borrows: usize,
    pub dups: usize,
    pub async_copies: usize,
    pub warnings: Vec<String>,
}

struct Slot {
    ty: Ty,
    live: bool,
    depth: usize,
}

struct Checker<'a> {
    prog: &'a Program,
    target: &'a Target,
    slots: HashMap<Var, Slot>,
    /// Values whose definition already failed. Uses of them are not
    /// reported again, so one mistake produces one diagnostic.
    poisoned: HashSet<Var>,
    ranges: HashMap<Var, Option<(i64, i64)>>,
    depth: usize,
    live_tg: usize,
    report: Report,
    diags: Vec<Diag>,
}

pub fn check(prog: &Program, target: &Target) -> Result<Report, Vec<Diag>> {
    let mut c = Checker {
        prog,
        target,
        slots: HashMap::new(),
        poisoned: HashSet::new(),
        ranges: HashMap::new(),
        depth: 0,
        live_tg: 0,
        report: Report::default(),
        diags: vec![],
    };
    c.block(&prog.body);
    let peak = c.report.peak_threadgroup_bytes;
    if peak > target.max_threadgroup_bytes {
        c.err(
            Kind::Budget,
            format!(
                "peak threadgroup footprint is {peak} B; {} allows {} B",
                target.name, target.max_threadgroup_bytes
            ),
        );
    }
    if c.diags.is_empty() {
        Ok(c.report)
    } else {
        Err(c.diags)
    }
}

impl Checker<'_> {
    fn err(&mut self, kind: Kind, msg: String) {
        self.diags.push(Diag { kind, msg });
    }

    fn warn(&mut self, msg: String) {
        if !self.report.warnings.contains(&msg) {
            self.report.warnings.push(msg);
        }
    }

    fn name(&self, v: Var) -> &str {
        self.prog.name_of(v)
    }

    fn bump(&mut self, bytes: isize) {
        self.live_tg = (self.live_tg as isize + bytes) as usize;
        self.report.peak_threadgroup_bytes = self.report.peak_threadgroup_bytes.max(self.live_tg);
    }

    fn define(&mut self, v: Var, ty: Ty, scope: &mut Vec<Var>) {
        self.bump(ty.threadgroup_bytes() as isize);
        self.slots.insert(
            v,
            Slot {
                ty,
                live: true,
                depth: self.depth,
            },
        );
        scope.push(v);
    }

    fn lookup(&mut self, v: Var) -> Option<&Slot> {
        if self.poisoned.contains(&v) {
            return None;
        }
        if !self.slots.contains_key(&v) {
            let n = self.name(v).to_string();
            self.err(Kind::Scope, format!("`{n}` is not in scope"));
            return None;
        }
        self.slots.get(&v)
    }

    fn consume(&mut self, v: Var) -> Option<Ty> {
        let depth = self.depth;
        let (ty, live, def_depth) = {
            let s = self.lookup(v)?;
            (s.ty.clone(), s.live, s.depth)
        };
        let n = self.name(v).to_string();
        if !ty.is_linear() {
            self.err(Kind::Type, format!("loop index `{n}` is not a value"));
            return None;
        }
        if !live {
            self.err(Kind::UseAfterMove, format!("`{n}` used after it was moved"));
            return None;
        }
        if def_depth < depth {
            self.err(
                Kind::ConsumeOuter,
                format!(
                    "`{n}` is defined outside this loop body and would be consumed on every \
                     iteration; borrow it, or carry it through the loop"
                ),
            );
            return None;
        }
        self.slots.get_mut(&v).expect("slot").live = false;
        self.bump(-(ty.threadgroup_bytes() as isize));
        self.report.moves += 1;
        Some(ty)
    }

    fn borrow(&mut self, v: Var) -> Option<Ty> {
        let (ty, live) = {
            let s = self.lookup(v)?;
            (s.ty.clone(), s.live)
        };
        let n = self.name(v).to_string();
        if !ty.is_linear() {
            self.err(Kind::Type, format!("loop index `{n}` is not a value"));
            return None;
        }
        if !live {
            self.err(
                Kind::UseAfterMove,
                format!("`{n}` borrowed after it was moved"),
            );
            return None;
        }
        self.report.borrows += 1;
        Some(ty)
    }

    fn range_of(&mut self, idx: Var) -> Option<Option<(i64, i64)>> {
        match self.slots.get(&idx).map(|s| &s.ty) {
            Some(Ty::Index) => Some(self.ranges.get(&idx).copied().flatten()),
            Some(_) => {
                let n = self.name(idx).to_string();
                self.err(
                    Kind::Type,
                    format!("`{n}` is used as an index but is a value"),
                );
                None
            }
            None => {
                if !self.poisoned.contains(&idx) {
                    let n = self.name(idx).to_string();
                    self.err(Kind::Scope, format!("index `{n}` is not in scope"));
                }
                None
            }
        }
    }

    /// Type every operand of one op. Borrows are resolved before moves, and a
    /// value may not be both in the same op.
    fn args(&mut self, args: &[Arg]) -> Option<Vec<Ty>> {
        for (i, a) in args.iter().enumerate() {
            if let Arg::Move(v) = a {
                let clash = args
                    .iter()
                    .enumerate()
                    .any(|(j, b)| j != i && b.var() == *v);
                if clash {
                    let n = self.name(*v).to_string();
                    self.err(
                        Kind::MovedWhileBorrowed,
                        format!("`{n}` is moved by an op that also uses it"),
                    );
                    return None;
                }
            }
        }
        let mut out: Vec<Option<Ty>> = vec![None; args.len()];
        for (i, a) in args.iter().enumerate() {
            match *a {
                Arg::Borrow(v) => out[i] = Some(self.borrow(v)?),
                Arg::BorrowElem(arr, idx) => {
                    let ty = self.borrow(arr)?;
                    let range = self.range_of(idx)?;
                    let Ty::Array(t, n) = ty else {
                        let an = self.name(arr).to_string();
                        self.err(Kind::Type, format!("`{an}` is indexed but is not an array"));
                        return None;
                    };
                    if let Some((lo, hi)) = range
                        && (lo < 0 || hi >= n as i64)
                    {
                        let an = self.name(arr).to_string();
                        self.err(
                            Kind::Bounds,
                            format!(
                                "index range {lo}..={hi} out of bounds for `{an}` of length {n}"
                            ),
                        );
                        return None;
                    }
                    out[i] = Some(Ty::Tile(t));
                }
                Arg::Move(_) => {}
            }
        }
        for (i, a) in args.iter().enumerate() {
            if let Arg::Move(v) = *a {
                out[i] = Some(self.consume(v)?);
            }
        }
        Some(out.into_iter().map(|t| t.expect("typed")).collect())
    }

    fn tile(&mut self, ty: Ty, what: &str) -> Option<TileTy> {
        match ty {
            Ty::Tile(t) => Some(t),
            Ty::Future(_) => {
                self.err(
                    Kind::Type,
                    format!("{what} is a future; `wait` on it before reading the tile"),
                );
                None
            }
            other => {
                self.err(
                    Kind::Type,
                    format!("{what} must be a tile, found {other:?}"),
                );
                None
            }
        }
    }

    fn view(&mut self, view: &View, writing: bool) -> Option<(DType, Vec<usize>)> {
        let Some(p) = self.prog.params.get(view.param).cloned() else {
            self.err(
                Kind::Scope,
                format!("parameter #{} does not exist", view.param),
            );
            return None;
        };
        if writing && !p.writable {
            self.err(Kind::Type, format!("parameter `{}` is read-only", p.name));
            return None;
        }
        if view.offset.len() != p.shape.len() || view.shape.len() != p.shape.len() {
            self.err(
                Kind::Shape,
                format!("view of `{}` has the wrong rank", p.name),
            );
            return None;
        }
        for d in 0..p.shape.len() {
            let (lo, hi) = self.expr_range(&view.offset[d])?;
            if lo < 0 || hi + view.shape[d] as i64 > p.shape[d] as i64 {
                self.err(
                    Kind::Bounds,
                    format!(
                        "view of `{}` dim {d} spans {lo}..{} over the loop, but the dim is {}",
                        p.name,
                        hi + view.shape[d] as i64,
                        p.shape[d]
                    ),
                );
                return None;
            }
        }
        Some((p.dtype, view.shape.clone()))
    }

    fn expr_range(&mut self, e: &IdxExpr) -> Option<(i64, i64)> {
        let (mut lo, mut hi) = (e.constant, e.constant);
        for &(v, c) in &e.terms {
            // An empty loop never evaluates the view; any range will do.
            let Some((a, b)) = self.range_of(v)? else {
                continue;
            };
            let (x, y) = (a * c, b * c);
            lo += x.min(y);
            hi += x.max(y);
        }
        Some((lo, hi))
    }

    fn not_narrower(&mut self, from: DType, to: DType, what: &str) -> bool {
        if to.size_bytes() < from.size_bytes() {
            self.err(
                Kind::Narrowing,
                format!("{what} narrows {from:?} to {to:?} implicitly; use `convert`"),
            );
            return false;
        }
        true
    }

    fn op(&mut self, op: &Op) -> Option<Option<Ty>> {
        let reg = |dtype: DType, shape: &[usize]| Ty::Tile(TileTy::new(dtype, shape, Space::Reg));
        Some(match op {
            Op::Alloc(t) | Op::Fill(t, _) => {
                if t.space == Space::Global {
                    self.err(Kind::Type, "tiles live in threadgroup or registers".into());
                    return None;
                }
                Some(Ty::Tile(t.clone()))
            }
            Op::Load(view, t) => {
                let (dt, shape) = self.view(view, false)?;
                if shape != t.shape {
                    self.err(
                        Kind::Shape,
                        format!("load of {shape:?} into a {:?} tile", t.shape),
                    );
                    return None;
                }
                if !self.not_narrower(dt, t.dtype, "load") {
                    return None;
                }
                Some(Ty::Tile(t.clone()))
            }
            Op::CopyAsync(view, dst) => {
                let (dt, shape) = self.view(view, false)?;
                let ty = self.consume(*dst)?;
                let t = self.tile(ty, "copy destination")?;
                if t.space != Space::Threadgroup {
                    self.err(Kind::Type, "async copies land in threadgroup memory".into());
                    return None;
                }
                if shape != t.shape || dt != t.dtype {
                    self.err(
                        Kind::Shape,
                        format!(
                            "copy of {dt:?}{shape:?} into {:?}{:?}; the copy engine does not convert",
                            t.dtype, t.shape
                        ),
                    );
                    return None;
                }
                self.report.async_copies += 1;
                if !self.target.async_copy {
                    self.warn(format!(
                        "{} has no async copy engine; copies lower to synchronous loads and the \
                         pipeline hides no latency",
                        self.target.name
                    ));
                }
                Some(Ty::Future(t))
            }
            Op::Wait(f) => match self.consume(*f)? {
                Ty::Future(t) => Some(Ty::Tile(t)),
                other => {
                    let n = self.name(*f).to_string();
                    self.err(Kind::Type, format!("wait on `{n}`, which is {other:?}"));
                    return None;
                }
            },
            Op::Store(a, view) => {
                let (dt, shape) = self.view(view, true)?;
                let ty = self.args(&[*a])?.remove(0);
                let t = self.tile(ty, "stored value")?;
                if t.shape != shape {
                    self.err(
                        Kind::Shape,
                        format!("store of {:?} into {shape:?}", t.shape),
                    );
                    return None;
                }
                if !self.not_narrower(t.dtype, dt, "store") {
                    return None;
                }
                if t.dtype != dt {
                    self.err(Kind::Type, format!("store of {:?} into {dt:?}", t.dtype));
                    return None;
                }
                None
            }
            Op::Drop(v) => {
                self.consume(*v)?;
                None
            }
            Op::Dup(a) => {
                let ty = self.args(&[*a])?.remove(0);
                let t = self.tile(ty, "dup operand")?;
                self.report.dups += 1;
                Some(Ty::Tile(t))
            }
            Op::MatMulNT(a, b, acc) | Op::MatMul(a, b, acc) => {
                let tys = self.args(&[*a, *b])?;
                let ta = self.tile(tys[0].clone(), "matmul lhs")?;
                let tb = self.tile(tys[1].clone(), "matmul rhs")?;
                if ta.shape.len() != 2 || tb.shape.len() != 2 {
                    self.err(Kind::Shape, "matmul operands must be 2-d".into());
                    return None;
                }
                let nt = matches!(op, Op::MatMulNT(..));
                let (m, k) = (ta.shape[0], ta.shape[1]);
                let (kb, n) = if nt {
                    (tb.shape[1], tb.shape[0])
                } else {
                    (tb.shape[0], tb.shape[1])
                };
                if k != kb {
                    self.err(
                        Kind::Shape,
                        format!(
                            "matmul {:?} x {:?}: inner dims disagree",
                            ta.shape, tb.shape
                        ),
                    );
                    return None;
                }
                if !self.not_narrower(ta.dtype, *acc, "matmul accumulator")
                    || !self.not_narrower(tb.dtype, *acc, "matmul accumulator")
                {
                    return None;
                }
                Some(reg(*acc, &[m, n]))
            }
            Op::Binary(bop, a, b) => {
                let tys = self.args(&[*a, *b])?;
                let ta = self.tile(tys[0].clone(), "lhs")?;
                let tb = self.tile(tys[1].clone(), "rhs")?;
                let row_bcast = ta.shape.len() == 2 && tb.shape == [ta.shape[0]];
                if ta.shape != tb.shape && !row_bcast {
                    self.err(
                        Kind::Shape,
                        format!("{} of {:?} and {:?}", bop.name(), ta.shape, tb.shape),
                    );
                    return None;
                }
                if ta.dtype != tb.dtype {
                    self.err(
                        Kind::Type,
                        format!("{} mixes {:?} and {:?}", bop.name(), ta.dtype, tb.dtype),
                    );
                    return None;
                }
                Some(reg(ta.dtype, &ta.shape))
            }
            Op::Exp(a) | Op::Scale(a, _) => {
                let ty = self.args(&[*a])?.remove(0);
                let t = self.tile(ty, "operand")?;
                Some(reg(t.dtype, &t.shape))
            }
            Op::RowReduce(_, a) => {
                let ty = self.args(&[*a])?.remove(0);
                let t = self.tile(ty, "reduce operand")?;
                if t.shape.len() != 2 {
                    self.err(Kind::Shape, "row reduce needs a 2-d tile".into());
                    return None;
                }
                Some(reg(t.dtype, &[t.shape[0]]))
            }
            Op::Convert(a, dt) => {
                let ty = self.args(&[*a])?.remove(0);
                let t = self.tile(ty, "convert operand")?;
                Some(reg(*dt, &t.shape))
            }
            Op::MakeArray(vs) => {
                if vs.is_empty() {
                    self.err(Kind::Shape, "empty array".into());
                    return None;
                }
                let mut elem: Option<TileTy> = None;
                for &v in vs {
                    let ty = self.consume(v)?;
                    let t = self.tile(ty, "array element")?;
                    match &elem {
                        None => elem = Some(t),
                        Some(e) if *e != t => {
                            self.err(Kind::Type, "array elements differ in type".into());
                            return None;
                        }
                        Some(_) => {}
                    }
                }
                Some(Ty::Array(elem.expect("nonempty"), vs.len()))
            }
        })
    }

    fn declared(&mut self, v: Var) -> Option<Ty> {
        let t = self.prog.declared[v.0 as usize].clone();
        if t.is_none() {
            let n = self.name(v).to_string();
            self.err(
                Kind::Type,
                format!("region parameter `{n}` has no declared type"),
            );
        }
        t
    }

    /// Check a block at the current depth; returns the types of its yields.
    /// Every linear value the block defines must be gone by its end.
    fn block(&mut self, b: &Block) -> Option<Vec<Ty>> {
        let mut scope = vec![];
        let mut ok = true;
        for s in &b.stmts {
            if self.stmt(s, &mut scope).is_none() {
                ok = false;
                self.poisoned.extend(defs(s));
            }
        }
        let mut yields = vec![];
        for &y in &b.yields {
            match self.consume(y) {
                Some(t) => yields.push(t),
                None => ok = false,
            }
        }
        // After an error, what looks like a leak is usually the error's
        // shadow; report leaks only in blocks that otherwise checked.
        let clean = ok;
        for v in scope.iter().filter(|_| clean) {
            let slot = &self.slots[v];
            if slot.live && slot.ty.is_linear() {
                let n = self.name(*v).to_string();
                self.err(
                    Kind::Leak,
                    format!("`{n}` is never consumed; drop it, yield it or move it into an op"),
                );
                ok = false;
            }
        }
        for v in &scope {
            if let Some(s) = self.slots.remove(v)
                && s.live
            {
                self.bump(-(s.ty.threadgroup_bytes() as isize));
            }
            self.ranges.remove(v);
        }
        ok.then_some(yields)
    }

    fn stmt(&mut self, s: &Stmt, scope: &mut Vec<Var>) -> Option<()> {
        match s {
            Stmt::Let { dst, op } => {
                let ty = self.op(op)?;
                match (dst, ty) {
                    (Some(v), Some(t)) => self.define(*v, t, scope),
                    (None, None) => {}
                    (Some(v), None) => {
                        let n = self.name(*v).to_string();
                        self.err(
                            Kind::Type,
                            format!("`{n}` is bound to an op with no result"),
                        );
                        return None;
                    }
                    (None, Some(_)) => {
                        self.err(Kind::Leak, "op result is discarded without a drop".into());
                        return None;
                    }
                }
            }
            Stmt::For {
                index,
                start,
                end,
                init,
                params,
                body,
                results,
            } => {
                if init.len() != params.len() || results.len() != params.len() {
                    self.err(Kind::Carry, "loop carry arity mismatch".into());
                    return None;
                }
                let mut carry = vec![];
                for (&i, &p) in init.iter().zip(params) {
                    let got = self.consume(i)?;
                    let want = self.declared(p)?;
                    if got != want {
                        let n = self.name(i).to_string();
                        self.err(
                            Kind::Carry,
                            format!("loop entered with `{n}`: {got:?}, carry declares {want:?}"),
                        );
                        return None;
                    }
                    carry.push(want);
                }
                let range = (start < end).then(|| (*start as i64, *end as i64 - 1));
                let yields = self.region(*index, range, params, &carry, body)?;
                if yields != carry {
                    for (k, (y, c)) in yields.iter().zip(&carry).enumerate() {
                        if y != c {
                            self.err(
                                Kind::Carry,
                                format!("loop yields {y:?} for carry #{k}, which was {c:?}"),
                            );
                        }
                    }
                    if yields.len() != carry.len() {
                        self.err(Kind::Carry, "loop yields the wrong number of values".into());
                    }
                    return None;
                }
                for (&r, t) in results.iter().zip(carry) {
                    self.define(r, t, scope);
                }
            }
            Stmt::MapEach {
                index,
                arrays,
                params,
                body,
                results,
            } => {
                if arrays.len() != params.len() {
                    self.err(Kind::Shape, "map_each arity mismatch".into());
                    return None;
                }
                let mut elems = vec![];
                let mut len = None;
                for (&a, &p) in arrays.iter().zip(params) {
                    let ty = self.consume(a)?;
                    let Ty::Array(t, n) = ty else {
                        let an = self.name(a).to_string();
                        self.err(Kind::Type, format!("map_each over `{an}`, not an array"));
                        return None;
                    };
                    if *len.get_or_insert(n) != n {
                        self.err(
                            Kind::Shape,
                            "map_each over arrays of different lengths".into(),
                        );
                        return None;
                    }
                    let want = self.declared(p)?;
                    if want != Ty::Tile(t.clone()) {
                        self.err(Kind::Type, "map_each element type mismatch".into());
                        return None;
                    }
                    elems.push(want);
                }
                let n = len.unwrap_or(0);
                // The other n-1 elements of every array still exist while one
                // is being worked on; they pin their storage.
                let rest: usize = elems
                    .iter()
                    .map(|t| t.threadgroup_bytes() * n.saturating_sub(1))
                    .sum();
                self.bump(rest as isize);
                let range = (n > 0).then(|| (0, n as i64 - 1));
                let yields = self.region(*index, range, params, &elems, body);
                self.bump(-(rest as isize));
                let yields = yields?;
                if yields.len() != results.len() {
                    self.err(
                        Kind::Carry,
                        "map_each yields the wrong number of values".into(),
                    );
                    return None;
                }
                for (&r, y) in results.iter().zip(yields) {
                    let t = self.tile(y, "map_each yield")?;
                    self.define(r, Ty::Array(t, n), scope);
                }
            }
        }
        Some(())
    }

    /// Enter a loop body one level deeper, with its index and parameters.
    fn region(
        &mut self,
        index: Var,
        range: Option<(i64, i64)>,
        params: &[Var],
        tys: &[Ty],
        body: &Block,
    ) -> Option<Vec<Ty>> {
        self.depth += 1;
        let mut scope = vec![];
        self.define(index, Ty::Index, &mut scope);
        self.ranges.insert(index, range);
        for (&p, t) in params.iter().zip(tys) {
            self.define(p, t.clone(), &mut scope);
        }
        // Parameters belong to the body's block: fold them into its scope so
        // the leak check covers them.
        let yields = self.block_with(body, scope);
        self.depth -= 1;
        yields
    }

    fn block_with(&mut self, b: &Block, pre: Vec<Var>) -> Option<Vec<Ty>> {
        // `block` owns its scope; seed it by checking params as part of it.
        let yields = self.block(b);
        let clean = yields.is_some();
        let mut ok = clean;
        for v in &pre {
            if let Some(s) = self.slots.remove(v) {
                if s.live && s.ty.is_linear() && clean {
                    let n = self.name(*v).to_string();
                    self.err(
                        Kind::Leak,
                        format!("loop parameter `{n}` is neither consumed nor yielded"),
                    );
                    ok = false;
                }
                if s.live {
                    self.bump(-(s.ty.threadgroup_bytes() as isize));
                }
            }
            self.ranges.remove(v);
        }
        if ok { yields } else { None }
    }
}

/// Values a statement defines in its enclosing block.
fn defs(s: &Stmt) -> Vec<Var> {
    match s {
        Stmt::Let { dst, .. } => dst.iter().copied().collect(),
        Stmt::For { results, .. } | Stmt::MapEach { results, .. } => results.clone(),
    }
}
