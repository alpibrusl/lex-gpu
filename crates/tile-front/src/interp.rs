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
//!
//! Roles of a warp-specialised region run on real threads, so their
//! interleaving is whatever the OS makes of it, and pipes block exactly as
//! barriers would. A share re-reads its slot on every borrow and fails if the
//! producer has refilled it — so a protocol bug shows up as a race, not as a
//! quietly different answer. A wait that cannot finish is reported as a
//! deadlock after [`RunOptions::deadlock_after`].

use std::collections::HashMap;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use half::f16;
use tile_ir::DType;

/// One NVFP4 value: sign, 2-bit exponent, 1-bit mantissa.
pub fn e2m1(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

/// One FP8 E4M3 byte (OCP `e4m3fn`: no infinities, max 448).
pub fn e4m3(b: u8) -> f32 {
    let (e, m) = ((b >> 3) & 0xF, (b & 7) as f32);
    let v = if e == 0 {
        m * 2.0f32.powi(-9)
    } else if e == 15 && m == 7.0 {
        f32::NAN
    } else {
        (1.0 + m / 8.0) * 2.0f32.powi(e as i32 - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}

use crate::ir::{
    Arg, Block, IdxExpr, Nibbles, Op, PipeDecl, Program, Reduce, Role, Stmt, TileTy, Var, View,
};

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
        DType::I8 => x.round().clamp(-128.0, 127.0),
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
    Producer(Var),
    Consumer(Var, usize),
    Slot {
        pipe: Var,
        slot: usize,
    },
    Share {
        pipe: Var,
        slot: usize,
        generation: u64,
    },
}

/// One pipe's ring, as the barriers would see it.
struct Ring {
    stages: usize,
    consumers: usize,
    data: Vec<Option<Vec<Tile>>>,
    /// Bumped on every commit into a slot; a share remembers the one it saw.
    generation: Vec<u64>,
    /// Releases the slot has had since it was last acquired. The slot is
    /// empty — acquirable — when this equals `consumers`.
    released: Vec<usize>,
    acquired: usize,
    /// Slot of the n-th commit. Consumer k's r-th receive reads `log[r]`.
    log: Vec<usize>,
    received: Vec<usize>,
}

#[derive(Default)]
struct SyncState {
    rings: HashMap<Var, Ring>,
    abort: Option<String>,
}

struct Shared {
    globals: Mutex<Vec<Tensor>>,
    sync: Mutex<SyncState>,
    cv: Condvar,
    deadlock_after: Duration,
}

#[derive(Clone, Copy, Debug)]
pub struct RunOptions {
    /// How long a pipe wait may block before the run is declared deadlocked.
    pub deadlock_after: Duration,
}

impl Default for RunOptions {
    fn default() -> RunOptions {
        RunOptions {
            deadlock_after: Duration::from_secs(10),
        }
    }
}

pub type Result<T> = std::result::Result<T, String>;

struct Interp<'a> {
    prog: &'a Program,
    shared: &'a Shared,
    env: HashMap<Var, Val>,
}

/// Run `prog` over `globals`, one tensor per parameter in declaration order.
/// Writable parameters are updated in place.
pub fn run(prog: &Program, globals: &mut [Tensor]) -> Result<()> {
    run_with(prog, globals, RunOptions::default())
}

/// Run a program that reads runtime scalars, one value per
/// `Program::dyn_scalars` entry, in order.
pub fn run_dyn(prog: &Program, globals: &mut [Tensor], scalars: &[u32]) -> Result<()> {
    run_full(prog, globals, scalars, RunOptions::default())
}

pub fn run_with(prog: &Program, globals: &mut [Tensor], opts: RunOptions) -> Result<()> {
    run_full(prog, globals, &[], opts)
}

fn run_full(
    prog: &Program,
    globals: &mut [Tensor],
    scalars: &[u32],
    opts: RunOptions,
) -> Result<()> {
    if scalars.len() != prog.dyn_scalars.len() {
        return Err(format!(
            "{} reads {} runtime scalars, got {}",
            prog.name,
            prog.dyn_scalars.len(),
            scalars.len()
        ));
    }
    for (&(v, max), &x) in prog.dyn_scalars.iter().zip(scalars) {
        if x as usize > max {
            return Err(format!(
                "scalar `{}` = {x} exceeds its bound {max}",
                prog.name_of(v)
            ));
        }
    }
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
    let shared = Shared {
        globals: Mutex::new(globals.to_vec()),
        sync: Mutex::new(SyncState::default()),
        cv: Condvar::new(),
        deadlock_after: opts.deadlock_after,
    };
    let mut it = Interp {
        prog,
        shared: &shared,
        env: HashMap::new(),
    };
    // Instances are independent by construction (disjoint writes are the
    // program's contract), so running them one after another is faithful.
    for (&(v, _), &x) in prog.dyn_scalars.iter().zip(scalars) {
        it.env.insert(v, Val::Index(x as i64));
    }
    for g2 in 0..prog.grid2 {
        if let Some(pid2) = prog.pid2 {
            it.env.insert(pid2, Val::Index(g2 as i64));
        }
        for g in 0..prog.grid {
            if let Some(pid) = prog.pid {
                it.env.insert(pid, Val::Index(g as i64));
            }
            it.block(&prog.body)?;
        }
    }
    globals.clone_from_slice(&shared.globals.into_inner().expect("globals lock"));
    Ok(())
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, SyncState> {
        self.sync.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn abort(&self, why: String) {
        let mut st = self.lock();
        st.abort.get_or_insert(why);
        self.cv.notify_all();
    }

    /// Block until `ready` holds, as a barrier wait would.
    fn wait_until<'g>(
        &'g self,
        mut st: MutexGuard<'g, SyncState>,
        what: &str,
        mut ready: impl FnMut(&SyncState) -> bool,
    ) -> Result<MutexGuard<'g, SyncState>> {
        let deadline = Instant::now() + self.deadlock_after;
        loop {
            if st.abort.is_some() {
                return Err("aborted: another role failed".into());
            }
            if ready(&st) {
                return Ok(st);
            }
            let now = Instant::now();
            if now >= deadline {
                let why = format!("deadlock: {what} waited {:?}", self.deadlock_after);
                st.abort = Some(why.clone());
                self.cv.notify_all();
                return Err(why);
            }
            st = self
                .cv
                .wait_timeout(st, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
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
        let p = &self.prog.params[view.param];
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
        let dims = &self.prog.params[param].shape;
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
        let idx = self.view_elems(param, offset, &ty.shape);
        let globals = self
            .shared
            .globals
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let src = &globals[param].data;
        let data = idx.into_iter().map(|i| round(ty.dtype, src[i])).collect();
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
            Arg::BorrowPart(share, part) => {
                let &Val::Share {
                    pipe,
                    slot,
                    generation,
                } = self.get(share)?
                else {
                    return Err(format!("`{}` is not a share", self.name(share)));
                };
                let st = self.shared.lock();
                let ring = st.rings.get(&pipe).ok_or("share of a closed pipe")?;
                if ring.generation[slot] != generation {
                    return Err(format!(
                        "race: slot {slot} was refilled while `{}` still shared it",
                        self.name(share)
                    ));
                }
                ring.data[slot].as_ref().ok_or("share of an empty slot")?[part].clone()
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
                let mut globals = self
                    .shared
                    .globals
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let g = &mut globals[view.param];
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
                let two = ta.ty.shape.len() == 2;
                let row = two && tb.ty.shape == [ta.ty.shape[0]];
                let col = two && !row && tb.ty.shape == [1, ta.ty.shape[1]];
                let cols = if two { ta.ty.shape[1] } else { 1 };
                let out = ta
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let y = if row {
                            tb.data[i / cols]
                        } else if col {
                            tb.data[i % cols]
                        } else {
                            tb.data[i]
                        };
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
            Op::Unary(u, a) => {
                let t = self.arg(*a)?;
                let out = t.data.iter().map(|&x| u.apply(x)).collect();
                Some(Val::Tile(self.fresh(t.ty.dtype, &t.ty.shape, out)))
            }
            Op::SwapPairs(a) => {
                let t = self.arg(*a)?;
                let out = (0..t.data.len()).map(|i| t.data[i ^ 1]).collect();
                Some(Val::Tile(self.fresh(t.ty.dtype, &t.ty.shape, out)))
            }
            Op::Dequant(q, s, m, group) => {
                let (tq, ts) = (self.arg(*q)?, self.arg(*s)?);
                let tm = m.map(|m| self.arg(m)).transpose()?;
                let c = tq.ty.shape[1];
                let out = tq
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let g = (i / c) * (c / group) + (i % c) / group;
                        x * ts.data[g] - tm.as_ref().map_or(0.0, |m| m.data[g])
                    })
                    .collect();
                Some(Val::Tile(self.fresh(DType::F32, &tq.ty.shape, out)))
            }
            Op::Dequant4(q, s, m, group, mode) => {
                let (tq, ts) = (self.arg(*q)?, self.arg(*s)?);
                let tm = m.map(|m| self.arg(m)).transpose()?;
                let (r, qc) = (tq.ty.shape[0], tq.ty.shape[1]);
                let pairs = *mode == Nibbles::Pairs;
                let c = if pairs { 2 * qc } else { qc };
                let out = (0..r * c)
                    .map(|i| {
                        let (row, col) = (i / c, i % c);
                        let qi = row * qc + if pairs { col / 2 } else { col };
                        let byte = (tq.data[qi] as i32) & 0xFF;
                        let high = match mode {
                            Nibbles::Low => false,
                            Nibbles::High => true,
                            Nibbles::Pairs => col % 2 == 1,
                        };
                        let nib = if high { byte >> 4 } else { byte & 0xF };
                        let g = row * (c / group) + col / group;
                        nib as f32 * ts.data[g] - tm.as_ref().map_or(0.0, |m| m.data[g])
                    })
                    .collect();
                Some(Val::Tile(self.fresh(DType::F32, &[r, c], out)))
            }
            Op::DequantFp4(q, sc, gs, group) => {
                let (tq, ts, tg) = (self.arg(*q)?, self.arg(*sc)?, self.arg(*gs)?);
                let (r, c) = (tq.ty.shape[0], 2 * tq.ty.shape[1]);
                let out = (0..r * c)
                    .map(|i| {
                        let (row, col) = (i / c, i % c);
                        let byte = (tq.data[row * c / 2 + col / 2] as i32) & 0xFF;
                        let code = if col % 2 == 0 { byte & 0xF } else { byte >> 4 };
                        let g = row * (c / group) + col / group;
                        let scale = e4m3(ts.data[g] as i32 as u8);
                        e2m1(code as u8) * scale * tg.data[row]
                    })
                    .collect();
                Some(Val::Tile(self.fresh(DType::F32, &[r, c], out)))
            }
            Op::Dequant6(lo, hi, s, group) => {
                let (tl, th, ts) = (self.arg(*lo)?, self.arg(*hi)?, self.arg(*s)?);
                let (r, c) = (tl.ty.shape[0], 2 * tl.ty.shape[1]);
                let out = (0..r * c)
                    .map(|i| {
                        let (row, col) = (i / c, i % c);
                        let lb = (tl.data[row * c / 2 + col / 2] as i32) & 0xFF;
                        let hb = (th.data[row * c / 4 + col / 4] as i32) & 0xFF;
                        let l = if col % 2 == 0 { lb & 0xF } else { lb >> 4 };
                        let h = (hb >> (2 * (col % 4))) & 3;
                        let q = (l | (h << 4)) - 32;
                        q as f32 * ts.data[row * (c / group) + col / group]
                    })
                    .collect();
                Some(Val::Tile(self.fresh(DType::F32, &[r, c], out)))
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
            Op::MaskCols(a, first, limit) => {
                let t = self.arg(*a)?;
                let (f, lim) = (self.eval(first)?, self.eval(limit)?);
                let n = t.ty.shape[1];
                let out = t
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        if f + (i % n) as i64 >= lim {
                            f32::NEG_INFINITY
                        } else {
                            x
                        }
                    })
                    .collect();
                Some(Val::Tile(self.fresh(t.ty.dtype, &t.ty.shape, out)))
            }
            Op::Convert(a, dt) => {
                let t = self.arg(*a)?;
                Some(Val::Tile(self.fresh(*dt, &t.ty.shape, t.data)))
            }
            Op::Acquire(h) => {
                let Val::Producer(pipe) = *self.get(*h)? else {
                    return Err("acquire through a non-producer".into());
                };
                let st = self.shared.lock();
                let mut st = self.shared.wait_until(st, "acquire", |st| {
                    let r = &st.rings[&pipe];
                    r.released[r.acquired % r.stages] == r.consumers
                })?;
                let r = st.rings.get_mut(&pipe).expect("ring");
                let slot = r.acquired % r.stages;
                r.released[slot] = 0;
                r.acquired += 1;
                drop(st);
                std::thread::yield_now();
                Some(Val::Slot { pipe, slot })
            }
            Op::Commit(h, views, slot) => {
                let Val::Producer(pipe) = *self.get(*h)? else {
                    return Err("commit through a non-producer".into());
                };
                let Val::Slot { pipe: sp, slot } = self.take(*slot)? else {
                    return Err("commit of a non-slot".into());
                };
                if sp != pipe {
                    return Err("commit into another pipe's slot".into());
                }
                let mut parts = vec![];
                for v in views {
                    let off = self.offsets(v)?;
                    let ty = TileTy::new(
                        self.prog.params[v.param].dtype,
                        &v.shape,
                        tile_ir::Space::Threadgroup,
                    );
                    parts.push(self.read_global(v.param, &off, &ty));
                }
                let mut st = self.shared.lock();
                let r = st.rings.get_mut(&pipe).expect("ring");
                r.data[slot] = Some(parts);
                r.generation[slot] += 1;
                r.log.push(slot);
                self.shared.cv.notify_all();
                drop(st);
                std::thread::yield_now();
                None
            }
            Op::Receive(h) => {
                let Val::Consumer(pipe, k) = *self.get(*h)? else {
                    return Err("receive through a non-consumer".into());
                };
                let st = self.shared.lock();
                let mut st = self.shared.wait_until(st, "receive", |st| {
                    let r = &st.rings[&pipe];
                    r.log.len() > r.received[k]
                })?;
                let r = st.rings.get_mut(&pipe).expect("ring");
                let slot = r.log[r.received[k]];
                r.received[k] += 1;
                let generation = r.generation[slot];
                drop(st);
                std::thread::yield_now();
                Some(Val::Share {
                    pipe,
                    slot,
                    generation,
                })
            }
            Op::Release(h, share) => {
                let Val::Consumer(pipe, _) = *self.get(*h)? else {
                    return Err("release through a non-consumer".into());
                };
                let Val::Share { pipe: sp, slot, .. } = self.take(*share)? else {
                    return Err("release of a non-share".into());
                };
                if sp != pipe {
                    return Err("release into another pipe".into());
                }
                let mut st = self.shared.lock();
                st.rings.get_mut(&pipe).expect("ring").released[slot] += 1;
                self.shared.cv.notify_all();
                drop(st);
                std::thread::yield_now();
                None
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
                end_dyn,
                init,
                params,
                body,
                results,
            } => {
                let mut carry: Vec<Val> =
                    init.iter().map(|&v| self.take(v)).collect::<Result<_>>()?;
                let stop = match end_dyn {
                    Some(d) => (self.index(*d)? as usize).min(*end),
                    None => *end,
                };
                for i in *start..stop {
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
            Stmt::Specialize { pipes, roles } => self.specialize(pipes, roles)?,
        }
        Ok(())
    }

    fn specialize(&mut self, pipes: &[PipeDecl], roles: &[Role]) -> Result<()> {
        {
            let mut st = self.shared.lock();
            for p in pipes {
                let t = &p.ty;
                st.rings.insert(
                    t.id,
                    Ring {
                        stages: t.stages,
                        consumers: t.consumers,
                        data: vec![None; t.stages],
                        generation: vec![0; t.stages],
                        released: vec![t.consumers; t.stages],
                        acquired: 0,
                        log: vec![],
                        received: vec![0; t.consumers],
                    },
                );
            }
        }
        for p in pipes {
            self.env.insert(p.ty.id, Val::Producer(p.ty.id));
            for (k, &c) in p.consumers.iter().enumerate() {
                self.env.insert(c, Val::Consumer(p.ty.id, k));
            }
        }
        let mut moved = vec![];
        for r in roles {
            let vals: Vec<Val> = r
                .inputs
                .iter()
                .map(|&i| self.take(i))
                .collect::<Result<_>>()?;
            moved.push(vals);
        }

        // Each role sees what it may borrow from the enclosing scope, plus
        // what was moved into it, and runs on its own thread.
        let (prog, shared, outer) = (self.prog, self.shared, &self.env);
        let results: Vec<Result<Vec<Val>>> = std::thread::scope(|sc| {
            let handles: Vec<_> = roles
                .iter()
                .zip(moved)
                .map(|(r, vals)| {
                    let mut env = outer.clone();
                    env.extend(r.params.iter().copied().zip(vals));
                    sc.spawn(move || {
                        let out = Interp { prog, shared, env }.block(&r.body);
                        if let Err(e) = &out {
                            shared.abort(format!("role `{}`: {e}", r.name));
                        }
                        out
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("role panicked".into())))
                .collect()
        });
        self.shared
            .lock()
            .rings
            .retain(|id, _| !pipes.iter().any(|p| p.ty.id == *id));

        // Report the root cause, not the roles that were aborted by it.
        if results.iter().any(Result::is_err) {
            let why = self.shared.lock().abort.clone();
            return Err(why.unwrap_or_else(|| "a role failed".into()));
        }
        for (r, out) in roles.iter().zip(results) {
            for (&v, val) in r.results.iter().zip(out?) {
                self.env.insert(v, val);
            }
        }
        Ok(())
    }
}
