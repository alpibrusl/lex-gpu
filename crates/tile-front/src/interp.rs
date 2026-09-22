//! Reference interpreter for tile programs.
//!
//! Arithmetic is f32; a value is rounded to its tile's dtype whenever a tile
//! of that dtype is produced, so an f16 tile holds exactly what an f16
//! register would. Moves really remove values from the environment, so a
//! program that slipped past the checker still fails loudly here instead of
//! computing with stale data.
//!
//! Copies are *issued* at `copy_async` (offsets evaluated with the loop
//! indices of that moment) and *land* at `wait`. Global inputs are read-only,
//! so this is observationally the same as copying eagerly — but it is the
//! semantics a hardware backend has, and it keeps the interpreter honest if
//! kernels ever read what they write.

use std::collections::HashMap;

use half::f16;
use tile_ir::DType;

use crate::ir::{Arg, Block, IdxExpr, Op, Program, Reduce, Stmt, TileTy, Var, View};

/// A global tensor. Values are held as f32, already rounded to `dtype`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(dtype: DType, shape: &[usize], data: &[f32]) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Tensor {
            dtype,
            shape: shape.to_vec(),
            data: data.iter().map(|&x| round(dtype, x)).collect(),
        }
    }

    pub fn zeros(dtype: DType, shape: &[usize]) -> Tensor {
        Tensor::new(dtype, shape, &vec![0.0; shape.iter().product()])
    }
}

pub fn round(dt: DType, x: f32) -> f32 {
    match dt {
        DType::F32 => x,
        DType::F16 => f16::from_f32(x).to_f32(),
    }
}

#[derive(Clone, Debug)]
struct Tile {
    ty: TileTy,
    data: Vec<f32>,
    init: bool,
}

#[derive(Clone, Debug)]
struct Pending {
    buf: Tile,
    param: usize,
    offset: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Val {
    Tile(Tile),
    Future(Box<Pending>),
    Array(Vec<Tile>),
    Index(i64),
}

pub type Result<T> = std::result::Result<T, String>;

struct Interp<'a> {
    prog: &'a Program,
    globals: &'a mut [Tensor],
    env: HashMap<Var, Val>,
}

/// Run `prog` over `globals`, one tensor per parameter in declaration order.
/// Writable parameters are updated in place.
pub fn run(prog: &Program, globals: &mut [Tensor]) -> Result<()> {
    if globals.len() != prog.params.len() {
        return Err(format!(
            "{} expects {} tensors, got {}",
            prog.name,
            prog.params.len(),
            globals.len()
        ));
    }
    for (p, g) in prog.params.iter().zip(globals.iter()) {
        if p.shape != g.shape || p.dtype != g.dtype {
            return Err(format!(
                "tensor for `{}` has the wrong shape or dtype",
                p.name
            ));
        }
    }
    let mut it = Interp {
        prog,
        globals,
        env: HashMap::new(),
    };
    it.block(&prog.body)?;
    Ok(())
}

impl Interp<'_> {
    fn name(&self, v: Var) -> &str {
        self.prog.name_of(v)
    }

    fn take(&mut self, v: Var) -> Result<Val> {
        self.env
            .remove(&v)
            .ok_or_else(|| format!("`{}` used after move", self.name(v)))
    }

    fn get(&self, v: Var) -> Result<&Val> {
        self.env
            .get(&v)
            .ok_or_else(|| format!("`{}` borrowed after move", self.name(v)))
    }

    fn index(&self, v: Var) -> Result<i64> {
        match self.get(v)? {
            Val::Index(i) => Ok(*i),
            _ => Err(format!("`{}` is not an index", self.name(v))),
        }
    }

    fn eval(&self, e: &IdxExpr) -> Result<i64> {
        let mut x = e.constant;
        for &(v, c) in &e.terms {
            x += c * self.index(v)?;
        }
        Ok(x)
    }

    fn offsets(&self, view: &View) -> Result<Vec<usize>> {
        let p = &self.globals[view.param];
        let mut out = vec![];
        for (d, e) in view.offset.iter().enumerate() {
            let o = self.eval(e)?;
            if o < 0 || o as usize + view.shape[d] > p.shape[d] {
                return Err(format!("view out of bounds on dim {d}: {o}"));
            }
            out.push(o as usize);
        }
        Ok(out)
    }

    /// Linear element offsets of a view, in row-major order of the view.
    fn view_elems(&self, param: usize, offset: &[usize], shape: &[usize]) -> Vec<usize> {
        let dims = &self.globals[param].shape;
        let n: usize = shape.iter().product();
        let mut out = Vec::with_capacity(n);
        for flat in 0..n {
            let (mut rem, mut lin) = (flat, 0);
            let mut coord = vec![0; shape.len()];
            for d in (0..shape.len()).rev() {
                coord[d] = rem % shape[d];
                rem /= shape[d];
            }
            for d in 0..shape.len() {
                lin = lin * dims[d] + offset[d] + coord[d];
            }
            out.push(lin);
        }
        out
    }

    fn read_global(&self, param: usize, offset: &[usize], ty: &TileTy) -> Tile {
        let src = &self.globals[param].data;
        let data = self
            .view_elems(param, offset, &ty.shape)
            .into_iter()
            .map(|i| round(ty.dtype, src[i]))
            .collect();
        Tile {
            ty: ty.clone(),
            data,
            init: true,
        }
    }

    /// Resolve an operand to a tile: moved out, or cloned out of a borrow.
    /// Cloning here is an interpreter convenience, not a semantic copy — the
    /// checker has already proven the borrow is read-only.
    fn arg(&mut self, a: Arg) -> Result<Tile> {
        let t = match a {
            Arg::Move(v) => match self.take(v)? {
                Val::Tile(t) => t,
                _ => return Err(format!("`{}` is not a tile", self.name(v))),
            },
            Arg::Borrow(v) => match self.get(v)? {
                Val::Tile(t) => t.clone(),
                _ => return Err(format!("`{}` is not a tile", self.name(v))),
            },
            Arg::BorrowElem(arr, idx) => {
                let i = self.index(idx)? as usize;
                match self.get(arr)? {
                    Val::Array(ts) => ts
                        .get(i)
                        .cloned()
                        .ok_or_else(|| format!("index {i} out of bounds"))?,
                    _ => return Err(format!("`{}` is not an array", self.name(arr))),
                }
            }
        };
        if !t.init {
            return Err(format!("read of an uninitialised `{}`", self.name(a.var())));
        }
        Ok(t)
    }

    fn fresh(&self, dtype: DType, shape: &[usize], data: Vec<f32>) -> Tile {
        Tile {
            ty: TileTy::new(dtype, shape, tile_ir::Space::Reg),
            data: data.into_iter().map(|x| round(dtype, x)).collect(),
            init: true,
        }
    }

    fn op(&mut self, op: &Op) -> Result<Option<Val>> {
        Ok(match op {
            Op::Alloc(t) => Some(Val::Tile(Tile {
                ty: t.clone(),
                data: vec![0.0; t.elems()],
                init: false,
            })),
            Op::Fill(t, x) => Some(Val::Tile(Tile {
                ty: t.clone(),
                data: vec![round(t.dtype, *x); t.elems()],
                init: true,
            })),
            Op::Load(view, t) => {
                let off = self.offsets(view)?;
                Some(Val::Tile(self.read_global(view.param, &off, t)))
            }
            Op::CopyAsync(view, dst) => {
                let offset = self.offsets(view)?;
                let buf = match self.take(*dst)? {
                    Val::Tile(t) => t,
                    _ => return Err("copy destination is not a tile".into()),
                };
                Some(Val::Future(Box::new(Pending {
                    buf,
                    param: view.param,
                    offset,
                })))
            }
            Op::Wait(f) => match self.take(*f)? {
                Val::Future(p) => Some(Val::Tile(self.read_global(p.param, &p.offset, &p.buf.ty))),
                _ => return Err(format!("wait on non-future `{}`", self.name(*f))),
            },
            Op::Store(a, view) => {
                let t = self.arg(*a)?;
                let off = self.offsets(view)?;
                let idx = self.view_elems(view.param, &off, &view.shape);
                let g = &mut self.globals[view.param];
                for (i, x) in idx.into_iter().zip(t.data) {
                    g.data[i] = round(g.dtype, x);
                }
                None
            }
            Op::Drop(v) => {
                self.take(*v)?;
                None
            }
            Op::Dup(a) => Some(Val::Tile(self.arg(*a)?)),
            Op::MatMulNT(a, b, acc) | Op::MatMul(a, b, acc) => {
                let (ta, tb) = (self.arg(*a)?, self.arg(*b)?);
                let (m, k) = (ta.ty.shape[0], ta.ty.shape[1]);
                let nt = matches!(op, Op::MatMulNT(..));
                let n = if nt { tb.ty.shape[0] } else { tb.ty.shape[1] };
                let mut out = vec![0.0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0.0f32;
                        for p in 0..k {
                            let bv = if nt {
                                tb.data[j * k + p]
                            } else {
                                tb.data[p * n + j]
                            };
                            s += ta.data[i * k + p] * bv;
                        }
                        out[i * n + j] = s;
                    }
                }
                Some(Val::Tile(self.fresh(*acc, &[m, n], out)))
            }
            Op::Binary(bop, a, b) => {
                let (ta, tb) = (self.arg(*a)?, self.arg(*b)?);
                let row = ta.ty.shape.len() == 2 && tb.ty.shape == [ta.ty.shape[0]];
                let cols = if row { ta.ty.shape[1] } else { 1 };
                let out = ta
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let y = if row { tb.data[i / cols] } else { tb.data[i] };
                        bop.apply(x, y)
                    })
                    .collect();
                Some(Val::Tile(self.fresh(ta.ty.dtype, &ta.ty.shape, out)))
            }
            Op::Exp(a) => {
                let t = self.arg(*a)?;
                let out = t.data.iter().map(|x| x.exp()).collect();
                Some(Val::Tile(self.fresh(t.ty.dtype, &t.ty.shape, out)))
            }
            Op::Scale(a, s) => {
                let t = self.arg(*a)?;
                let out = t.data.iter().map(|x| x * s).collect();
                Some(Val::Tile(self.fresh(t.ty.dtype, &t.ty.shape, out)))
            }
            Op::RowReduce(r, a) => {
                let t = self.arg(*a)?;
                let (m, n) = (t.ty.shape[0], t.ty.shape[1]);
                let out = (0..m)
                    .map(|i| {
                        let row = &t.data[i * n..(i + 1) * n];
                        match r {
                            Reduce::Max => row.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                            Reduce::Sum => row.iter().sum(),
                        }
                    })
                    .collect();
                Some(Val::Tile(self.fresh(t.ty.dtype, &[m], out)))
            }
            Op::Convert(a, dt) => {
                let t = self.arg(*a)?;
                Some(Val::Tile(self.fresh(*dt, &t.ty.shape, t.data)))
            }
            Op::MakeArray(vs) => {
                let mut ts = vec![];
                for &v in vs {
                    match self.take(v)? {
                        Val::Tile(t) => ts.push(t),
                        _ => return Err("array element is not a tile".into()),
                    }
                }
                Some(Val::Array(ts))
            }
        })
    }

    fn block(&mut self, b: &Block) -> Result<Vec<Val>> {
        for s in &b.stmts {
            self.stmt(s)?;
        }
        b.yields.iter().map(|&y| self.take(y)).collect()
    }

    fn stmt(&mut self, s: &Stmt) -> Result<()> {
        match s {
            Stmt::Let { dst, op } => {
                let v = self.op(op)?;
                if let (Some(d), Some(v)) = (dst, v) {
                    self.env.insert(*d, v);
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
                let mut carry: Vec<Val> =
                    init.iter().map(|&v| self.take(v)).collect::<Result<_>>()?;
                for i in *start..*end {
                    self.env.insert(*index, Val::Index(i as i64));
                    for (&p, v) in params.iter().zip(carry) {
                        self.env.insert(p, v);
                    }
                    carry = self.block(body)?;
                }
                self.env.remove(index);
                for (&r, v) in results.iter().zip(carry) {
                    self.env.insert(r, v);
                }
            }
            Stmt::MapEach {
                index,
                arrays,
                params,
                body,
                results,
            } => {
                let mut ins: Vec<std::vec::IntoIter<Tile>> = vec![];
                for &a in arrays {
                    match self.take(a)? {
                        Val::Array(ts) => ins.push(ts.into_iter()),
                        _ => return Err(format!("`{}` is not an array", self.name(a))),
                    }
                }
                let n = ins.first().map_or(0, |it| it.len());
                let mut outs: Vec<Vec<Tile>> = vec![vec![]; results.len()];
                for i in 0..n {
                    self.env.insert(*index, Val::Index(i as i64));
                    for (&p, it) in params.iter().zip(ins.iter_mut()) {
                        let t = it.next().ok_or("ragged map_each")?;
                        self.env.insert(p, Val::Tile(t));
                    }
                    for (o, y) in outs.iter_mut().zip(self.block(body)?) {
                        match y {
                            Val::Tile(t) => o.push(t),
                            _ => return Err("map_each yielded a non-tile".into()),
                        }
                    }
                }
                self.env.remove(index);
                for (&r, ts) in results.iter().zip(outs) {
                    self.env.insert(r, Val::Array(ts));
                }
            }
        }
        Ok(())
    }
}
