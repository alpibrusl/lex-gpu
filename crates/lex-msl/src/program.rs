//! Lower a checked `lex-front` program to one MSL kernel.
//!
//! P2's contract is *correct at any speed*, so this is the simplest lowering
//! that is right for every program the checker accepts:
//!
//! - One threadgroup per grid instance. `T` threads, chosen by the caller.
//! - **Register tiles** are distributed cyclically: thread `t` owns flat
//!   elements `t, t + T, t + 2T, …`, held in a `PER = ceil(n / T)` array.
//!   Elementwise ops between equal-sized tiles never leave the thread.
//! - An operand some *other* thread owns — a matmul input, a row-broadcast
//!   vector, a row reduction's input — is **staged** through a threadgroup
//!   scratch area first.
//! - **Threadgroup tiles** get a fixed offset in a static arena per allocation
//!   site, and travel through copies, waits and loop carries as pointers. That
//!   is what makes a rotated double buffer a pointer swap.
//! - Metal has no async copy engine, so `copy_async` is a cooperative copy at
//!   issue time and `wait` is a barrier. That is only sound because linearity
//!   proves nothing still owns the destination buffer.
//! - Barriers are conservative: before every threadgroup write, and between
//!   staging and use. Deriving the minimal set from the effect graph is P3.
//! - **Loads from read-only parameters are lazy.** A register-tile load
//!   becomes an address expression, not a copy; `convert` and `dequant` over
//!   lazy operands stay lazy. A consumer reads any element straight from
//!   device memory, so nothing is staged: a quantised matvec reads each
//!   weight once, dequantises it inline, and feeds the reduction. That is
//!   operator fusion, decided here rather than written by hand. A lazy tile
//!   is materialised only where storage is required (loop carries, arrays).
//!
//! Everything is computed in f32 and narrowed only where a lex's dtype says
//! so, which is exactly what the interpreter does — so the GPU and the
//! interpreter differ only in `exp`'s last bits.

use std::collections::HashMap;
use std::fmt::Write as _;

use lex_front::ir::{
    Arg, BinOp, Block, IdxExpr, Nibbles, Op, Program, Reduce, Stmt, TileTy, UnOp, Var, View,
};
use lex_ir::{DType, Space, Target};

use crate::dialect::{Dialect, Msl, Param};

/// A lowered kernel and everything needed to launch it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lowered {
    pub entry: String,
    pub source: String,
    /// Threadgroups to launch: one per grid instance, along x…
    pub grid: usize,
    /// …and along y (1 unless the program has a second grid dimension).
    pub grid2: usize,
    pub threads: usize,
    /// Arena (threadgroup tiles) plus staging scratch.
    pub threadgroup_bytes: usize,
    pub arena_bytes: usize,
    pub scratch_bytes: usize,
    /// Barrier sites in the source (not dynamic count).
    pub barriers: usize,
    /// Per buffer binding, in order: does the kernel write it? Parameters
    /// then (if any) the runtime-scalar buffer, which is read-only. A
    /// runtime uses this to order dispatches only where they conflict.
    pub writes: Vec<bool>,
}

/// Where a value lives in the generated code.
#[derive(Clone, Debug)]
enum Loc {
    /// A distributed register tile: `expr[k]` is this thread's k-th element.
    Reg(String, TileTy),
    /// An array of register tiles: `name[i][k]`.
    RegArr(String, TileTy, usize),
    /// A threadgroup tile: `name[e]` is flat element `e`.
    Tg(String, TileTy),
    Index(String),
    /// A tile not held anywhere: an MSL expression for element `@I@`,
    /// evaluated where it is read (a load from a read-only parameter, or
    /// a conversion or dequantisation of one).
    Lazy(String, TileTy),
}

/// The index placeholder in a lazy lex's expression.
const AT: &str = "@I@";

fn at_index(template: &str, idx: &str) -> String {
    template.replace(AT, &format!("({idx})"))
}

/// How an op reads one operand at a flat element index.
enum Access {
    /// This thread's own element: `expr[k]`.
    Local(String),
    /// Any element, from threadgroup memory: `base[..]`.
    Shared(String),
    /// Any element, computed from device memory where it is read.
    Lazy(String),
}

struct Gen<'a> {
    prog: &'a Program,
    /// How this target spells barriers and lane shuffles. See
    /// [`crate::dialect`] for why this is a seam and not a second file.
    d: &'a dyn Dialect,
    threads: usize,
    body: String,
    depth: usize,
    locs: HashMap<Var, Loc>,
    arena: usize,
    scratch: usize,
    /// Scratch floats the current op's operands were staged into.
    staged: usize,
    barriers: usize,
    /// Lazy dequantisations, by result: the per-element value and the
    /// per-group scale and min as separate expressions, so a reduction over
    /// them can evaluate the group terms once per run instead of per value.
    dq: HashMap<Var, Dq>,
    /// The kernel decodes NVFP4, so the source needs the decode tables.
    fp4: bool,
}

/// A lazy dequantisation in parts: `v(I) * s(G) - m(G)` with
/// `G = (I / cols) * (cols / group) + (I % cols) / group`.
#[derive(Clone, Debug)]
struct Dq {
    v: String,
    s: String,
    m: Option<String>,
    /// A factor that depends only on the row (NVFP4's per-tensor scale),
    /// templated on the row index. A reduction lifts it out of the run
    /// loop: the compiler cannot, because a device load may alias the
    /// kernel's own stores.
    row: Option<String>,
    cols: usize,
    group: usize,
    /// How values are packed, so a reduction can load each byte once for
    /// all the values in it instead of once per value.
    pack: Pack,
}

#[derive(Clone, Debug)]
enum Pack {
    /// One value per element of the value tile.
    None,
    /// 4-bit pairs: the byte lex's expression and row length in bytes.
    Pairs(String, usize),
    /// 6-bit values in a 4-bit plane and a 2-bit plane: each plane's
    /// expression and row length in bytes.
    Six(String, usize, String, usize),
    /// NVFP4 pairs: the byte lex's expression and row length in bytes.
    /// Like [`Pack::Pairs`], but a nibble is an E2M1 float code.
    Fp4(String, usize),
}

/// Lower `prog` (which must already pass `lex_front::check` for `target`)
/// with `threads` threads per threadgroup.
pub fn lower(prog: &Program, target: &Target, threads: usize) -> Result<Lowered, String> {
    lower_with(prog, target, threads, &Msl)
}

/// Lower for a dialect other than Metal.
///
/// The whole point of the seam: the same 1,800 lines below decide what to
/// emit, and the dialect decides how to spell it. What is *not* yet behind
/// the seam is the NVFP4 decode preamble, so a program that dequantises
/// four-bit weights still emits Metal helpers and will not compile
/// elsewhere. Everything else does.
pub fn lower_with(
    prog: &Program,
    target: &Target,
    threads: usize,
    dialect: &dyn Dialect,
) -> Result<Lowered, String> {
    if threads == 0
        || !threads.is_multiple_of(target.simd_width)
        || threads > target.max_threads_per_threadgroup
    {
        return Err(format!(
            "{threads} threads is not a whole number of simdgroups within {}'s limit of {}",
            target.name, target.max_threads_per_threadgroup
        ));
    }
    let mut g = Gen {
        prog,
        d: dialect,
        threads,
        body: String::new(),
        depth: 1,
        locs: HashMap::new(),
        arena: 0,
        scratch: 0,
        staged: 0,
        barriers: 0,
        dq: HashMap::new(),
        fp4: false,
    };
    if let Some(pid) = prog.pid {
        g.locs.insert(pid, Loc::Index("gid".into()));
    }
    if let Some(pid2) = prog.pid2 {
        g.locs.insert(pid2, Loc::Index("gid2".into()));
    }
    for (i, &(x, _)) in prog.dyn_scalars.iter().enumerate() {
        g.locs.insert(x, Loc::Index(format!("scalars[{i}]")));
    }
    g.block(&prog.body)?;

    let entry = ident(&prog.name);
    let scratch_bytes = g.scratch * 4;
    let arena_bytes = g.arena;
    let threadgroup_bytes = arena_bytes + scratch_bytes;
    if threadgroup_bytes > target.max_threadgroup_bytes {
        return Err(format!(
            "lowering needs {threadgroup_bytes} B of threadgroup memory ({arena_bytes} B of tiles \
             + {scratch_bytes} B of staging scratch); {} allows {}",
            target.name, target.max_threadgroup_bytes
        ));
    }

    let mut s = String::new();
    let _ = writeln!(
        s,
        "// Generated by lex-msl from a lex-front program. Do not edit by hand."
    );
    let _ = writeln!(s, "// kernel : {}", prog.name);
    let _ = writeln!(s, "// target : {}", target.name);
    let _ = writeln!(
        s,
        "// launch : {} threadgroups x {threads} threads",
        prog.grid
    );
    let _ = writeln!(
        s,
        "// tg mem : {arena_bytes} B tiles + {scratch_bytes} B scratch of {} B",
        target.max_threadgroup_bytes
    );
    s.push_str(&g.d.includes());
    if g.fp4 {
        s.push_str(&g.d.fp4_preamble());
    }
    let names: Vec<String> = prog
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| param_ident(i, &p.name))
        .collect();
    let params: Vec<Param<'_>> = prog
        .params
        .iter()
        .zip(&names)
        .map(|(p, name)| Param {
            ty: g.d.scalar(p.dtype),
            name,
            writable: p.writable,
        })
        .collect();
    s.push_str(&g.d.entry(&entry, &params, !prog.dyn_scalars.is_empty()));
    if arena_bytes > 0 {
        let _ = writeln!(
            s,
            "    {}\n    {p} arena = ({p})arena4;",
            g.d.shared_array("float4", "arena4", arena_bytes.div_ceil(16)),
            p = g.d.shared_ptr("uchar")
        );
    }
    if g.scratch > 0 {
        let _ = writeln!(s, "    {}", g.d.shared_array("float", "scratch", g.scratch));
    }
    if g.fp4 {
        // The sixteen E2M1 values, one per lane of the simdgroup, read
        // once. A code then costs a shuffle instead of a load: with real
        // weights the indices scatter, and a constant-memory gather runs
        // at about half the bandwidth an arithmetic decode does.
        s.push_str("    const float fp4_lane = FP4_V[tid & 15u];\n");
    }
    s.push_str(&g.body);
    s.push_str("}\n");

    Ok(Lowered {
        entry,
        source: s,
        grid: prog.grid,
        grid2: prog.grid2,
        threads,
        threadgroup_bytes,
        arena_bytes,
        scratch_bytes,
        barriers: g.barriers,
        writes: prog
            .params
            .iter()
            .map(|p| p.writable)
            .chain((!prog.dyn_scalars.is_empty()).then_some(false))
            .collect(),
    })
}

fn ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn param_ident(i: usize, name: &str) -> String {
    format!("p{i}_{}", ident(name))
}

fn v(var: Var) -> String {
    format!("v{}", var.0)
}

/// Consecutive elements per lane per step in a split-K reduction: the
/// largest of `LEX_VEC` (else `default`), 8, 4, 2 that tiles the reduction
/// evenly. Measured on an M4 Max: 16 for packed weights (a byte holds two or
/// four values, so a longer run covers the same bytes), 8 otherwise.
fn split_k_vec(kd: usize, lanes: usize, default: usize) -> usize {
    let want: usize = std::env::var("LEX_VEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default);
    [want, 8, 4, 2]
        .into_iter()
        .find(|&v| v <= want && v > 0 && kd.is_multiple_of(lanes * v))
        .unwrap_or(1)
}

/// The storage name a location reads from.
fn storage(l: &Loc) -> String {
    match l {
        Loc::Reg(n, _) | Loc::RegArr(n, _, _) | Loc::Tg(n, _) | Loc::Index(n) => n.clone(),
        Loc::Lazy(e, _) => e.clone(),
    }
}

fn lit(x: f32) -> String {
    if x == f32::INFINITY {
        "INFINITY".into()
    } else if x == f32::NEG_INFINITY {
        "(-INFINITY)".into()
    } else {
        format!("{x:?}f")
    }
}

impl Gen<'_> {
    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.body.push_str("    ");
        }
        self.body.push_str(text);
        self.body.push('\n');
    }

    fn barrier(&mut self) {
        self.barriers += 1;
        let text = self.d.barrier();
        self.line(&text);
    }

    fn per(&self, n: usize) -> usize {
        n.div_ceil(self.threads)
    }

    fn loc(&self, x: Var) -> Result<Loc, String> {
        self.locs
            .get(&x)
            .cloned()
            .ok_or_else(|| format!("`{}` has no storage", self.prog.name_of(x)))
    }

    fn idx(&self, e: &IdxExpr) -> Result<String, String> {
        let mut parts = vec![e.constant.to_string()];
        for &(x, c) in &e.terms {
            let Loc::Index(name) = self.loc(x)? else {
                return Err("index expression over a non-index".into());
            };
            parts.push(format!("{c} * {name}"));
        }
        Ok(format!("(uint)({})", parts.join(" + ")))
    }

    /// Device address of flat element `e` of `view`, as MSL.
    fn addr(&self, view: &View, e: &str) -> Result<String, String> {
        let dims = &self.prog.params[view.param].shape;
        let r = dims.len();
        // A view spanning every inner dimension is one contiguous run of
        // memory: element `e` sits at `base + e`. Written that way, the
        // compiler can see that consecutive elements are adjacent and merge
        // their loads; the per-dimension `/` and `%` form hides it. Byte
        // offsets inside a packed run are written run-relative
        // (`p0 / 2 + u / 2`, `p0` a multiple of the run) for the same reason.
        if (1..r).all(|d| view.shape[d] == dims[d]) {
            let mut base = vec![];
            for d in 0..r {
                let stride: usize = dims[d + 1..].iter().product();
                base.push(format!("{} * {stride}u", self.idx(&view.offset[d])?));
            }
            return Ok(format!(
                "{}[{} + ({e})]",
                param_ident(view.param, &self.prog.params[view.param].name),
                base.join(" + ")
            ));
        }
        let mut terms = vec![];
        for d in 0..r {
            let inner: usize = view.shape[d + 1..].iter().product();
            let stride: usize = dims[d + 1..].iter().product();
            let coord = format!("(({e}) / {inner}u % {}u)", view.shape[d]);
            terms.push(format!(
                "({} + {coord}) * {stride}u",
                self.idx(&view.offset[d])?
            ));
        }
        Ok(format!(
            "{}[{}]",
            param_ident(view.param, &self.prog.params[view.param].name),
            terms.join(" + ")
        ))
    }

    /// `for each element this thread owns: body(k, e)`.
    fn owned(&mut self, n: usize, body: &[String]) {
        let (per, t) = (self.per(n), self.threads);
        self.line(&format!("for (uint k = 0; k < {per}u; ++k) {{"));
        self.depth += 1;
        self.line(&format!("const uint e = k * {t}u + tid;"));
        self.line(&format!("if (e < {n}u) {{"));
        self.depth += 1;
        for b in body {
            self.line(b);
        }
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
    }

    /// `for every element, cooperatively: body(e)`.
    fn every(&mut self, n: usize, body: &str) {
        let t = self.threads;
        self.line(&format!("for (uint e = tid; e < {n}u; e += {t}u) {{"));
        self.depth += 1;
        self.line(body);
        self.depth -= 1;
        self.line("}");
    }

    fn declare_reg(&mut self, x: Var, ty: &TileTy) -> String {
        let name = v(x);
        let per = self.per(ty.elems());
        self.line(&format!("{} {name}[{per}];", self.d.scalar(ty.dtype)));
        self.locs.insert(x, Loc::Reg(name.clone(), ty.clone()));
        name
    }

    fn declare_tg(&mut self, x: Var, ty: &TileTy) -> String {
        let name = v(x);
        let off = self.arena.next_multiple_of(16);
        self.arena = off + ty.bytes();
        let st = self.d.scalar(ty.dtype);
        self.line(&format!(
            "{p} {name} = ({p})(arena + {off});",
            p = self.d.shared_ptr(st)
        ));
        self.locs.insert(x, Loc::Tg(name.clone(), ty.clone()));
        name
    }

    /// Resolve operands for an op whose output has `out_elems` elements.
    /// `local[i]` says operand i is read at the output's own element index.
    /// Register operands that are not local are staged into scratch.
    fn operands(
        &mut self,
        args: &[Arg],
        local: &[bool],
        out_elems: usize,
    ) -> Result<Vec<(Access, TileTy)>, String> {
        let mut out = vec![];
        let mut stage = vec![];
        let mut off = 0;
        for (a, &loc_ok) in args.iter().zip(local) {
            let (expr, ty, reg) = match *a {
                Arg::Move(x) | Arg::Borrow(x) => match self.loc(x)? {
                    Loc::Reg(n, t) => (n, t, true),
                    Loc::Tg(n, t) => (n, t, false),
                    Loc::Lazy(e, t) => {
                        out.push((Access::Lazy(e), t));
                        continue;
                    }
                    _ => return Err("operand is not a tile".into()),
                },
                Arg::BorrowElem(arr, i) => {
                    let (Loc::RegArr(n, t, _), Loc::Index(iname)) = (self.loc(arr)?, self.loc(i)?)
                    else {
                        return Err("unsupported array element operand".into());
                    };
                    (format!("{n}[{iname}]"), t, true)
                }
                Arg::BorrowPart(..) => {
                    return Err("pipe shares have no Metal lowering".into());
                }
            };
            if !reg {
                out.push((Access::Shared(expr), ty));
            } else if loc_ok && ty.elems() == out_elems {
                out.push((Access::Local(expr), ty));
            } else {
                let base = format!("(scratch + {off})");
                stage.push((expr, ty.elems(), off));
                off += ty.elems();
                out.push((Access::Shared(base), ty));
            }
        }
        self.staged = off;
        if !stage.is_empty() {
            self.scratch = self.scratch.max(off);
            // Nobody may still be reading the scratch from the previous op.
            self.barrier();
            for (expr, n, o) in stage {
                self.owned(n, &[format!("scratch[{o} + e] = float({expr}[k]);")]);
            }
            self.barrier();
        }
        Ok(out)
    }

    fn read(a: &Access, at: &str) -> String {
        match a {
            Access::Local(x) => format!("float({x}[k])"),
            Access::Shared(x) => format!("float({x}[{at}])"),
            Access::Lazy(t) => format!("float({})", at_index(t, at)),
        }
    }

    fn block(&mut self, b: &Block) -> Result<(), String> {
        for s in &b.stmts {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match s {
            Stmt::Let { dst, op } => self.op(*dst, op),
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
                for (&i, &p) in init.iter().zip(params) {
                    let src = self.loc(i)?;
                    self.bind_copy(p, &src)?;
                }
                let iname = v(*index);
                let stop = match end_dyn {
                    Some(d) => {
                        let Loc::Index(dn) = self.loc(*d)? else {
                            return Err("dynamic loop end is not a scalar".into());
                        };
                        format!("min({end}u, {dn})")
                    }
                    None => format!("{end}u"),
                };
                self.line(&format!(
                    "for (uint {iname} = {start}u; {iname} < {stop}; ++{iname}) {{"
                ));
                self.depth += 1;
                self.locs.insert(*index, Loc::Index(iname));
                self.block(body)?;
                // Carries are assigned in parallel: a rotated double buffer
                // yields pointers that *are* other carries' storage, so a
                // sequential assignment would read one it just overwrote.
                let mut moves = vec![];
                for (&y, &p) in body.yields.iter().zip(params) {
                    moves.push((self.loc(p)?, self.loc(y)?));
                }
                let names: Vec<String> = moves.iter().map(|(d, _)| storage(d)).collect();
                let mut staged = vec![];
                for (i, (dst, src)) in moves.iter().enumerate() {
                    let aliases_other = names
                        .iter()
                        .enumerate()
                        .any(|(j, n)| j != i && *n == storage(src));
                    match (dst, src) {
                        (Loc::Tg(_, t), Loc::Tg(sn, _)) if aliases_other => {
                            let tmp = format!("{}_next", storage(dst));
                            self.line(&format!(
                                "{} {tmp} = {sn};",
                                self.d.shared_ptr(self.d.scalar(t.dtype))
                            ));
                            staged.push((dst.clone(), Loc::Tg(tmp, t.clone())));
                        }
                        _ if aliases_other => {
                            return Err(
                                "a loop that permutes register carries is not lowered yet".into()
                            );
                        }
                        _ => staged.push((dst.clone(), src.clone())),
                    }
                }
                for (dst, src) in staged {
                    self.assign(&dst, &src)?;
                }
                self.depth -= 1;
                self.line("}");
                for (&r, &p) in results.iter().zip(params) {
                    let l = self.loc(p)?;
                    self.locs.insert(r, l);
                }
                Ok(())
            }
            Stmt::MapEach {
                index,
                arrays,
                params,
                body,
                results,
            } => {
                let mut n = 0;
                let mut ins = vec![];
                for &a in arrays {
                    let Loc::RegArr(name, t, len) = self.loc(a)? else {
                        return Err("map_each over a non-register array is not lowered".into());
                    };
                    n = len;
                    ins.push((name, t));
                }
                // Result arrays live outside the loop, so they are declared
                // before the body that determines their element type. They
                // take the input arrays' element types, and the yields are
                // checked against that below.
                let mut outs = vec![];
                for (k, &r) in results.iter().enumerate() {
                    let t = ins
                        .get(k)
                        .map(|x| x.1.clone())
                        .ok_or("map_each yields more arrays than it walks")?;
                    let name = v(r);
                    let per = self.per(t.elems());
                    self.line(&format!("{} {name}[{n}][{per}];", self.d.scalar(t.dtype)));
                    outs.push((r, name, t));
                }
                let iname = v(*index);
                self.line(&format!(
                    "for (uint {iname} = 0; {iname} < {n}u; ++{iname}) {{"
                ));
                self.depth += 1;
                self.locs.insert(*index, Loc::Index(iname.clone()));
                for (&p, (name, t)) in params.iter().zip(&ins) {
                    let pname = v(p);
                    self.line(&format!(
                        "thread {}* {pname} = {name}[{iname}];",
                        self.d.scalar(t.dtype)
                    ));
                    self.locs.insert(p, Loc::Reg(pname, t.clone()));
                }
                self.block(body)?;
                for (&y, (_, name, t)) in body.yields.iter().zip(&outs) {
                    let src = self.loc(y)?;
                    if !matches!(&src, Loc::Reg(_, st) if st == t) {
                        return Err("map_each that changes element type is not lowered yet".into());
                    }
                    let dst = Loc::Reg(format!("{name}[{iname}]"), t.clone());
                    self.assign(&dst, &src)?;
                }
                self.depth -= 1;
                self.line("}");
                for (r, name, t) in outs {
                    self.locs.insert(r, Loc::RegArr(name, t, n));
                }
                Ok(())
            }
            Stmt::Specialize { .. } => {
                Err("warp-specialised roles have no Metal lowering (no split barriers)".into())
            }
        }
    }

    /// Declare fresh storage for `p` holding a copy of `src`.
    fn bind_copy(&mut self, p: Var, src: &Loc) -> Result<(), String> {
        match src {
            Loc::Reg(_, t) => {
                let t = t.clone();
                self.declare_reg(p, &t);
            }
            Loc::RegArr(_, t, n) => {
                let name = v(p);
                let per = self.per(t.elems());
                self.line(&format!("{} {name}[{n}][{per}];", self.d.scalar(t.dtype)));
                self.locs.insert(p, Loc::RegArr(name, t.clone(), *n));
            }
            Loc::Tg(_, t) => {
                let name = v(p);
                self.line(&format!(
                    "{} {name};",
                    self.d.shared_ptr(self.d.scalar(t.dtype))
                ));
                self.locs.insert(p, Loc::Tg(name, t.clone()));
            }
            Loc::Lazy(_, t) => {
                let t = t.clone();
                self.declare_reg(p, &t);
            }
            Loc::Index(_) => return Err("an index cannot be carried".into()),
        }
        let dst = self.loc(p)?;
        self.assign(&dst, src)
    }

    /// `dst = src`, by value for registers and by pointer for threadgroup.
    fn assign(&mut self, dst: &Loc, src: &Loc) -> Result<(), String> {
        match (dst, src) {
            (Loc::Reg(d, t), Loc::Reg(s, _)) => {
                if d != s {
                    let per = self.per(t.elems());
                    self.line(&format!(
                        "for (uint k = 0; k < {per}u; ++k) {d}[k] = {s}[k];"
                    ));
                }
            }
            (Loc::RegArr(d, t, n), Loc::RegArr(s, _, _)) => {
                if d != s {
                    let per = self.per(t.elems());
                    self.line(&format!(
                        "for (uint i = 0; i < {n}u; ++i) for (uint k = 0; k < {per}u; ++k) \
                         {d}[i][k] = {s}[i][k];"
                    ));
                }
            }
            (Loc::Tg(d, _), Loc::Tg(s, _)) => {
                if d != s {
                    self.line(&format!("{d} = {s};"));
                }
            }
            // A lazy value needs storage now: evaluate the elements each
            // thread owns.
            (Loc::Reg(d, t), Loc::Lazy(e, _)) => {
                let (d, t, e) = (d.clone(), t.clone(), e.clone());
                self.owned(
                    t.elems(),
                    &[format!(
                        "{d}[k] = {}({});",
                        self.d.scalar(t.dtype),
                        at_index(&e, "e")
                    )],
                );
            }
            _ => return Err("carry changes storage class".into()),
        }
        Ok(())
    }

    fn op(&mut self, dst: Option<Var>, op: &Op) -> Result<(), String> {
        let reg = |dt: DType, shape: &[usize]| TileTy::new(dt, shape, Space::Reg);
        match op {
            Op::Alloc(t) => {
                let x = dst.ok_or("alloc without a result")?;
                match t.space {
                    Space::Threadgroup => {
                        self.declare_tg(x, t);
                    }
                    _ => {
                        self.declare_reg(x, t);
                    }
                }
            }
            Op::Fill(t, val) => {
                let x = dst.ok_or("fill without a result")?;
                let st = self.d.scalar(t.dtype);
                match t.space {
                    Space::Threadgroup => {
                        let name = self.declare_tg(x, t);
                        self.barrier();
                        self.every(t.elems(), &format!("{name}[e] = {st}({});", lit(*val)));
                        self.barrier();
                    }
                    _ => {
                        let name = self.declare_reg(x, t);
                        self.owned(t.elems(), &[format!("{name}[k] = {st}({});", lit(*val))]);
                    }
                }
            }
            Op::Load(view, t) => {
                let x = dst.ok_or("load without a result")?;
                let st = self.d.scalar(t.dtype);
                let src = self.addr(view, "e")?;
                match t.space {
                    Space::Threadgroup => {
                        let name = self.declare_tg(x, t);
                        self.barrier();
                        self.every(t.elems(), &format!("{name}[e] = {st}({src});"));
                        self.barrier();
                    }
                    _ if !self.prog.params[view.param].writable
                        && std::env::var_os("LEX_NO_LAZY").is_none() =>
                    {
                        let e = format!("{st}({})", self.addr(view, AT)?);
                        self.locs.insert(x, Loc::Lazy(e, t.clone()));
                    }
                    _ => {
                        let name = self.declare_reg(x, t);
                        self.owned(t.elems(), &[format!("{name}[k] = {st}({src});")]);
                    }
                }
            }
            Op::CopyAsync(view, buf) => {
                let x = dst.ok_or("copy without a result")?;
                let Loc::Tg(name, t) = self.loc(*buf)? else {
                    return Err("copy destination is not threadgroup memory".into());
                };
                let src = self.addr(view, "e")?;
                // The destination was last read as some earlier tile: every
                // thread must be done with it before anyone overwrites it.
                self.barrier();
                self.every(t.elems(), &format!("{name}[e] = {src};"));
                self.locs.insert(x, Loc::Tg(name, t));
            }
            Op::Wait(f) => {
                let x = dst.ok_or("wait without a result")?;
                let l = self.loc(*f)?;
                self.barrier();
                self.locs.insert(x, l);
            }
            Op::Store(a, view) => {
                let dt = self.d.scalar(self.prog.params[view.param].dtype);
                let n = view.shape.iter().product();
                let ops = self.operands(&[*a], &[true], n)?;
                let target = self.addr(view, "e")?;
                match &ops[0].0 {
                    Access::Local(_) | Access::Lazy(_) => {
                        let val = Self::read(&ops[0].0, "e");
                        self.owned(n, &[format!("{target} = {dt}({val});")]);
                    }
                    Access::Shared(_) => {
                        let val = Self::read(&ops[0].0, "e");
                        self.barrier();
                        self.every(n, &format!("{target} = {dt}({val});"));
                    }
                }
            }
            Op::Drop(_) => {}
            Op::Binary(bop, a, b)
                if self.lazy(*a).is_some()
                    && self.lazy(*b).is_some()
                    && self.arg_ty(*a)?.shape == self.arg_ty(*b)?.shape =>
            {
                let x = dst.ok_or("op without a result")?;
                let (ea, t) = self.lazy(*a).expect("checked");
                let (eb, _) = self.lazy(*b).expect("checked");
                let (l, r) = (format!("float({ea})"), format!("float({eb})"));
                let e = match bop {
                    BinOp::Add => format!("({l} + {r})"),
                    BinOp::Sub => format!("({l} - {r})"),
                    BinOp::Mul => format!("({l} * {r})"),
                    BinOp::Div => format!("({l} / {r})"),
                    BinOp::Max => format!("max({l}, {r})"),
                };
                let e = format!("{}({e})", self.d.scalar(t.dtype));
                self.locs.insert(x, Loc::Lazy(e, reg(t.dtype, &t.shape)));
            }
            Op::Convert(a, dt) if self.lazy(*a).is_some() => {
                let x = dst.ok_or("op without a result")?;
                let (e, t) = self.lazy(*a).expect("checked");
                let e = self.d.convert(*dt, &format!("float({e})"));
                self.locs.insert(x, Loc::Lazy(e, reg(*dt, &t.shape)));
            }
            Op::Dequant(q, s, m, group) | Op::Dequant4(q, s, m, group, _)
                if [Some(*q), Some(*s), *m]
                    .iter()
                    .flatten()
                    .all(|a| self.lazy(*a).is_some()) =>
            {
                let x = dst.ok_or("op without a result")?;
                let (qe, tq) = self.lazy(*q).expect("checked");
                let (se, _) = self.lazy(*s).expect("checked");
                let qc = tq.shape[1];
                let pairs = matches!(op, Op::Dequant4(.., Nibbles::Pairs));
                let c = if pairs { 2 * qc } else { qc };
                let gidx = format!("({AT} / {c}u) * {}u + ({AT} % {c}u) / {group}u", c / group);
                let qv = match op {
                    Op::Dequant4(.., Nibbles::Pairs) => {
                        // Output element I lives in byte I/2 of its row,
                        // low nibble for even columns.
                        let byte =
                            at_index(&qe, &format!("({AT} / {c}u) * {qc}u + ({AT} % {c}u) / 2u"));
                        format!("float(((uint)(uchar)({byte}) >> (({AT} & 1u) * 4u)) & 0xFu)")
                    }
                    Op::Dequant4(.., mode) => format!(
                        "float(((uint)(uchar)({qe}) >> {}u) & 0xFu)",
                        if *mode == Nibbles::High { 4 } else { 0 }
                    ),
                    _ => format!("float({qe})"),
                };
                let sv = format!("float({})", at_index(&se, &gidx));
                let mv = match m {
                    Some(m) => {
                        let (me, _) = self.lazy(*m).expect("checked");
                        format!(" - float({})", at_index(&me, &gidx))
                    }
                    None => String::new(),
                };
                self.dq.insert(
                    x,
                    Dq {
                        v: qv.clone(),
                        s: format!("float({se})"),
                        m: m.map(|m| format!("float({})", self.lazy(m).expect("checked").0)),
                        cols: c,
                        group: *group,
                        row: None,
                        pack: if pairs {
                            Pack::Pairs(qe.clone(), qc)
                        } else {
                            Pack::None
                        },
                    },
                );
                let e = format!("({qv} * {sv}{mv})");
                self.locs
                    .insert(x, Loc::Lazy(e, reg(DType::F32, &[tq.shape[0], c])));
            }
            Op::DequantFp4(q, sc, gs, group)
                if [*q, *sc, *gs].iter().all(|a| self.lazy(*a).is_some()) =>
            {
                let x = dst.ok_or("op without a result")?;
                let (qe, tq) = self.lazy(*q).expect("checked");
                let (se, _) = self.lazy(*sc).expect("checked");
                let (ge, _) = self.lazy(*gs).expect("checked");
                let (r, c) = (tq.shape[0], 2 * tq.shape[1]);
                let per_row = c / group;
                self.fp4 = true;
                // Element I: byte I/2 of its row, low nibble for even
                // columns; scale group I/group; row scale I/c.
                let byte = at_index(
                    &qe,
                    &format!("({AT} / {c}u) * {}u + ({AT} % {c}u) / 2u", tq.shape[1]),
                );
                let v = self.d.shuffle(
                    "fp4_lane",
                    &format!("((uint)(uchar)({byte}) >> (({AT} & 1u) * 4u)) & 0xFu"),
                );
                // Inside a reduction the scale is formed once per group, so
                // the row scale rides along with it: group G is row
                // `G / per_row`.
                let s_of_group = format!("fp8_e4m3((uint)(uchar)({se}) & 0xFFu)");
                let row_of_group = at_index(&ge, &format!("({AT}) / {per_row}u"));
                let gidx = format!("({AT} / {c}u) * {per_row}u + ({AT} % {c}u) / {group}u");
                let e = format!(
                    "({v} * {} * float({}))",
                    at_index(&s_of_group, &gidx),
                    at_index(&row_of_group, &gidx)
                );
                self.dq.insert(
                    x,
                    Dq {
                        v,
                        s: s_of_group,
                        m: None,
                        cols: c,
                        group: *group,
                        row: Some(ge.clone()),
                        pack: Pack::Fp4(qe.clone(), tq.shape[1]),
                    },
                );
                self.locs.insert(x, Loc::Lazy(e, reg(DType::F32, &[r, c])));
            }
            Op::Dequant6(lo, hi, sc, group)
                if [*lo, *hi, *sc].iter().all(|a| self.lazy(*a).is_some()) =>
            {
                let x = dst.ok_or("op without a result")?;
                let (le, tl) = self.lazy(*lo).expect("checked");
                let (he, th) = self.lazy(*hi).expect("checked");
                let (se, _) = self.lazy(*sc).expect("checked");
                let (lc, hc) = (tl.shape[1], th.shape[1]);
                let c = 2 * lc;
                let lb = at_index(&le, &format!("({AT} / {c}u) * {lc}u + ({AT} % {c}u) / 2u"));
                let hb = at_index(&he, &format!("({AT} / {c}u) * {hc}u + ({AT} % {c}u) / 4u"));
                let qv = format!(
                    "float(int((((uint)(uchar)({lb}) >> (({AT} & 1u) * 4u)) & 0xFu) | \
                     ((((uint)(uchar)({hb}) >> (({AT} & 3u) * 2u)) & 3u) << 4u)) - 32)"
                );
                let gidx = format!("({AT} / {c}u) * {}u + ({AT} % {c}u) / {group}u", c / group);
                let sv = format!("float({})", at_index(&se, &gidx));
                self.dq.insert(
                    x,
                    Dq {
                        v: qv.clone(),
                        s: format!("float({se})"),
                        m: None,
                        cols: c,
                        group: *group,
                        row: None,
                        pack: Pack::Six(le.clone(), lc, he.clone(), hc),
                    },
                );
                let e = format!("({qv} * {sv})");
                self.locs
                    .insert(x, Loc::Lazy(e, reg(DType::F32, &[tl.shape[0], c])));
            }
            Op::Dequant6(lo, hi, sc, group) => {
                let x = dst.ok_or("op without a result")?;
                let tl = self.arg_ty(*lo)?;
                let (r, c) = (tl.shape[0], 2 * tl.shape[1]);
                let n = r * c;
                let ops = self.operands(&[*lo, *hi, *sc], &[false, false, false], n)?;
                let lb = Self::read(
                    &ops[0].0,
                    &format!("(e / {c}u) * {}u + (e % {c}u) / 2u", c / 2),
                );
                let hb = Self::read(
                    &ops[1].0,
                    &format!("(e / {c}u) * {}u + (e % {c}u) / 4u", c / 4),
                );
                let sv = Self::read(
                    &ops[2].0,
                    &format!("(e / {c}u) * {}u + (e % {c}u) / {group}u", c / group),
                );
                let name = self.declare_reg(x, &reg(DType::F32, &[r, c]));
                self.owned(
                    n,
                    &[format!(
                        "{name}[k] = float(int((((uint)(uchar)({lb}) >> ((e & 1u) * 4u)) & 0xFu) | \
                         ((((uint)(uchar)({hb}) >> ((e & 3u) * 2u)) & 3u) << 4u)) - 32) * {sv};"
                    )],
                );
            }
            Op::Dup(a) | Op::Exp(a) | Op::Scale(a, _) | Op::Convert(a, _) | Op::Unary(_, a) => {
                let x = dst.ok_or("op without a result")?;
                let ty = self.arg_ty(*a)?;
                let n = ty.elems();
                let ops = self.operands(&[*a], &[true], n)?;
                let src = Self::read(&ops[0].0, "e");
                let (dt, expr) = match op {
                    Op::Dup(_) => (ty.dtype, src),
                    Op::Exp(_) => (ty.dtype, self.d.exp(&src)),
                    Op::Scale(_, f) => (ty.dtype, format!("{src} * {}", lit(*f))),
                    Op::Convert(_, d) => (*d, src),
                    Op::Unary(UnOp::Rsqrt, _) => (ty.dtype, self.d.rsqrt(&src)),
                    Op::Unary(UnOp::Sigmoid, _) => (
                        ty.dtype,
                        format!("1.0f / (1.0f + {})", self.d.exp(&format!("-{src}"))),
                    ),
                    Op::Unary(UnOp::Softplus, _) => (
                        ty.dtype,
                        format!(
                            "{} + {}",
                            self.d.fmax(&src, "0.0f"),
                            self.d.log(&format!(
                                "1.0f + {}",
                                self.d.exp(&format!("-{}", self.d.fabs(&src)))
                            ))
                        ),
                    ),
                    _ => unreachable!(),
                };
                let name = self.declare_reg(x, &reg(dt, &ty.shape));
                let cv = self.d.convert(dt, &expr);
                self.owned(n, &[format!("{name}[k] = {cv};")]);
            }
            Op::Binary(bop, a, b) => {
                let x = dst.ok_or("op without a result")?;
                let (ta, tb) = (self.arg_ty(*a)?, self.arg_ty(*b)?);
                let n = ta.elems();
                let bcast = ta.shape != tb.shape;
                let col = bcast && tb.shape == [1, ta.shape[1]];
                let ops = self.operands(&[*a, *b], &[true, !bcast], n)?;
                let cols = if bcast { ta.shape[1] } else { 1 };
                let at = if col {
                    format!("e % {cols}u")
                } else {
                    format!("e / {cols}u")
                };
                let lhs = Self::read(&ops[0].0, "e");
                let rhs = Self::read(&ops[1].0, &at);
                let expr = match bop {
                    BinOp::Add => format!("{lhs} + {rhs}"),
                    BinOp::Sub => format!("{lhs} - {rhs}"),
                    BinOp::Mul => format!("{lhs} * {rhs}"),
                    BinOp::Div => format!("{lhs} / {rhs}"),
                    BinOp::Max => format!("max({lhs}, {rhs})"),
                };
                let dt = ta.dtype;
                let name = self.declare_reg(x, &reg(dt, &ta.shape));
                let cv = self.d.convert(dt, &expr);
                self.owned(n, &[format!("{name}[k] = {cv};")]);
            }
            Op::MaskCols(a, first, limit) => {
                let x = dst.ok_or("op without a result")?;
                let ty = self.arg_ty(*a)?;
                let (n, cols) = (ty.elems(), ty.shape[1]);
                let ops = self.operands(&[*a], &[true], n)?;
                let src = Self::read(&ops[0].0, "e");
                let lim = self.idx(limit)?;
                let f = self.idx(first)?;
                let name = self.declare_reg(x, &reg(ty.dtype, &ty.shape));
                self.owned(
                    n,
                    &[format!(
                        "{name}[k] = {}(({f} + e % {cols}u) >= {lim} ? -INFINITY : {src});",
                        self.d.scalar(ty.dtype)
                    )],
                );
            }
            Op::SwapPairs(a) => {
                let x = dst.ok_or("op without a result")?;
                let ty = self.arg_ty(*a)?;
                let n = ty.elems();
                let ops = self.operands(&[*a], &[false], n)?;
                let src = Self::read(&ops[0].0, "e ^ 1u");
                let name = self.declare_reg(x, &reg(ty.dtype, &ty.shape));
                self.owned(
                    n,
                    &[format!("{name}[k] = {}({src});", self.d.scalar(ty.dtype))],
                );
            }
            Op::Dequant(q, s, m, group) => {
                let x = dst.ok_or("op without a result")?;
                let tq = self.arg_ty(*q)?;
                let (n, c) = (tq.elems(), tq.shape[1]);
                let mut args = vec![*q, *s];
                args.extend(*m);
                let local: Vec<bool> = (0..args.len()).map(|i| i == 0).collect();
                let ops = self.operands(&args, &local, n)?;
                let at = format!("(e / {c}u) * {}u + (e % {c}u) / {group}u", c / group);
                let qv = Self::read(&ops[0].0, "e");
                let sv = Self::read(&ops[1].0, &at);
                let mv = ops
                    .get(2)
                    .map_or(String::new(), |o| format!(" - {}", Self::read(&o.0, &at)));
                let name = self.declare_reg(x, &reg(DType::F32, &tq.shape));
                self.owned(n, &[format!("{name}[k] = {qv} * {sv}{mv};")]);
            }
            Op::DequantFp4(q, sc, gs, group) => {
                let x = dst.ok_or("op without a result")?;
                let tq = self.arg_ty(*q)?;
                let (r, qc) = (tq.shape[0], tq.shape[1]);
                let (c, n) = (2 * qc, r * 2 * qc);
                self.fp4 = true;
                let ops = self.operands(&[*q, *sc, *gs], &[false, false, false], n)?;
                let byte = Self::read(&ops[0].0, &format!("(e / {c}u) * {qc}u + (e % {c}u) / 2u"));
                let at = format!("(e / {c}u) * {}u + (e % {c}u) / {group}u", c / group);
                let sv = Self::read(&ops[1].0, &at);
                let gv = Self::read(&ops[2].0, &format!("e / {c}u"));
                let name = self.declare_reg(x, &reg(DType::F32, &[r, c]));
                self.owned(
                    n,
                    &[format!(
                        "{name}[k] = {} \
                         * fp8_e4m3((uint)(uchar)({sv}) & 0xFFu) * {gv};",
                        self.d.shuffle(
                            "fp4_lane",
                            &format!("((uint)(uchar)({byte}) >> ((e & 1u) * 4u)) & 0xFu")
                        )
                    )],
                );
            }
            Op::Dequant4(q, s, m, group, mode) => {
                let x = dst.ok_or("op without a result")?;
                let tq = self.arg_ty(*q)?;
                let qc = tq.shape[1];
                let pairs = *mode == Nibbles::Pairs;
                let c = if pairs { 2 * qc } else { qc };
                let n = tq.shape[0] * c;
                let mut args = vec![*q, *s];
                args.extend(*m);
                // In pairs mode element e reads byte e/2: not the thread's
                // own element, so `q` is read shared (staged if in registers).
                let local: Vec<bool> = (0..args.len()).map(|i| i == 0 && !pairs).collect();
                let ops = self.operands(&args, &local, n)?;
                let at = format!("(e / {c}u) * {}u + (e % {c}u) / {group}u", c / group);
                let (byte, shift) = if pairs {
                    (
                        Self::read(&ops[0].0, &format!("(e / {c}u) * {qc}u + (e % {c}u) / 2u")),
                        "((e & 1u) * 4u)".to_string(),
                    )
                } else {
                    let s = if *mode == Nibbles::High { "4u" } else { "0u" };
                    (Self::read(&ops[0].0, "e"), s.to_string())
                };
                let qv = format!("float(((uint)(uchar)({byte}) >> {shift}) & 0xFu)");
                let sv = Self::read(&ops[1].0, &at);
                let mv = ops
                    .get(2)
                    .map_or(String::new(), |o| format!(" - {}", Self::read(&o.0, &at)));
                let name = self.declare_reg(x, &reg(DType::F32, &[tq.shape[0], c]));
                self.owned(n, &[format!("{name}[k] = {qv} * {sv}{mv};")]);
            }
            Op::MatMulNT(a, b, acc) | Op::MatMul(a, b, acc) => {
                let x = dst.ok_or("matmul without a result")?;
                let (ta, tb) = (self.arg_ty(*a)?, self.arg_ty(*b)?);
                let nt = matches!(op, Op::MatMulNT(..));
                let (m, kd) = (ta.shape[0], ta.shape[1]);
                let n = if nt { tb.shape[0] } else { tb.shape[1] };
                let ops = self.operands(&[*a, *b], &[false, false], m * n)?;
                let av = Self::read(&ops[0].0, &format!("i * {kd}u + p"));
                let bv = if nt {
                    Self::read(&ops[1].0, &format!("j * {kd}u + p"))
                } else {
                    Self::read(&ops[1].0, &format!("p * {n}u + j"))
                };
                // A few rows against a weight matrix (a small batch of
                // tokens): each lane group owns one weight row and keeps an
                // accumulator per token, so every weight is read and
                // dequantised once for the whole batch.
                // Weights from device memory only: attention's tiles already
                // live in threadgroup memory and keep the split-K path.
                let weight = matches!(b, Arg::Move(v) | Arg::Borrow(v)
                    if matches!(self.locs.get(v), Some(Loc::Lazy(..))));
                if nt && weight && m > 1 && m <= 16 && n < self.threads {
                    self.batched_rows(x, *acc, (m, n, kd), b, &ops)?;
                    return Ok(());
                }
                let outs = m * n;
                // Fewer outputs than threads (a matrix-vector product): give
                // each output a power-of-two group of lanes within one
                // simdgroup, split the reduction across them, and combine
                // with shuffles. Otherwise one thread per output.
                let mut lanes = 1;
                while lanes * 2 <= self.threads / outs && lanes * 2 <= 32 {
                    lanes *= 2;
                }
                if lanes >= 2 {
                    let res = self.staged;
                    self.scratch = self.scratch.max(res + outs);
                    // The partial sums go to scratch. With nothing staged
                    // (lazy operands), no staging barrier protects it, and a
                    // previous op's readers — the last chunk's result, say —
                    // may still be reading it: write-after-read. Wait.
                    if self.staged == 0 {
                        self.barrier();
                    }
                    self.line("{");
                    self.depth += 1;
                    self.line(&format!(
                        "const uint o = tid / {lanes}u, lane = tid % {lanes}u;"
                    ));
                    self.line("float s = 0.0f;");
                    self.line(&format!("if (o < {outs}u) {{"));
                    self.depth += 1;
                    self.line(&format!("const uint i = o / {n}u, j = o % {n}u;"));
                    // Each lane takes runs of `vec` consecutive elements, so
                    // the compiler can merge a run's loads into wide ones.
                    // B is a lazy dequantisation whose groups each run fits
                    // inside: form the group's scale and min once per run.
                    let dq = match b {
                        Arg::Move(v) | Arg::Borrow(v) if nt => self.dq.get(v).cloned(),
                        _ => None,
                    };
                    let packed = dq.as_ref().is_some_and(|d| !matches!(d.pack, Pack::None));
                    let vec = split_k_vec(kd, lanes, if packed { 16 } else { 8 });
                    let dq = dq.filter(|d| {
                        vec > 1
                            && d.cols == kd
                            && d.group.is_multiple_of(vec)
                            && match d.pack {
                                Pack::None => true,
                                Pack::Pairs(..) | Pack::Fp4(..) => vec.is_multiple_of(2),
                                Pack::Six(..) => vec.is_multiple_of(4),
                            }
                    });
                    if let Some(d) = dq {
                        let g = d.group;
                        let vv = at_index(&d.v, &format!("j * {kd}u + p"));
                        // Row-only factors are read once, not once per run.
                        let rs = match &d.row {
                            Some(r) => {
                                self.line(&format!(
                                    "const float rs = float({});",
                                    at_index(r, "j")
                                ));
                                " * rs"
                            }
                            None => "",
                        };
                        self.line(&format!(
                            "for (uint p0 = lane * {vec}u; p0 < {kd}u; p0 += {}u) {{",
                            lanes * vec
                        ));
                        self.line(&format!(
                            "    const uint grp = j * {}u + p0 / {g}u;",
                            kd / g
                        ));
                        self.line(&format!(
                            "    const float sg = {}{rs};",
                            at_index(&d.s, "grp")
                        ));
                        let mg = match &d.m {
                            Some(m) => {
                                self.line(&format!("    const float mg = {};", at_index(m, "grp")));
                                " - mg"
                            }
                            None => "",
                        };
                        match &d.pack {
                            Pack::Six(lo, lc, hi, hc) => {
                                // Four values per step: two low-plane bytes,
                                // one high-plane byte.
                                let b0 = at_index(lo, &format!("j * {lc}u + p0 / 2u + u / 2u"));
                                let b1 =
                                    at_index(lo, &format!("j * {lc}u + p0 / 2u + u / 2u + 1u"));
                                let h = at_index(hi, &format!("j * {hc}u + p0 / 4u + u / 4u"));
                                let a = |k: usize| {
                                    Self::read(&ops[0].0, &format!("i * {kd}u + p + {k}u"))
                                };
                                let (a0, a1, a2, a3) = (a(0), a(1), a(2), a(3));
                                self.line(&format!(
                                    "    for (uint u = 0; u < {vec}u; u += 4u) {{ const uint p = p0 + u; \
                                     const uint l0 = (uint)(uchar)({b0}), l1 = (uint)(uchar)({b1}), \
                                     hh = (uint)(uchar)({h}); \
                                     s += {a0} * (float(int((l0 & 0xFu) | ((hh & 3u) << 4u)) - 32) * sg) \
                                     + {a1} * (float(int((l0 >> 4u) | (((hh >> 2u) & 3u) << 4u)) - 32) * sg) \
                                     + {a2} * (float(int((l1 & 0xFu) | (((hh >> 4u) & 3u) << 4u)) - 32) * sg) \
                                     + {a3} * (float(int((l1 >> 4u) | (((hh >> 6u) & 3u) << 4u)) - 32) * sg); }}"
                                ));
                            }
                            Pack::Fp4(qe, qc) => {
                                // One byte, two E2M1 codes: p and p + 1.
                                // A run shares one scale, so it accumulates
                                // unscaled and pays a single multiply at
                                // the end rather than one per value: this
                                // loop is short of ALU, not of bandwidth.
                                // `fp4_pair` leaves its values 2^14 small,
                                // so the run's one multiply carries the
                                // 2^14 back and the decode stays free.
                                let byte = at_index(qe, &format!("j * {qc}u + p0 / 2u + u / 2u"));
                                let a1 = Self::read(&ops[0].0, &format!("i * {kd}u + p + 1u"));
                                self.line("    float run = 0.0f;");
                                self.line(&format!(
                                    "    for (uint u = 0; u < {vec}u; u += 2u) {{ const uint p = p0 + u; \
                                     const uint bq = (uint)(uchar)({byte}); \
                                     const float2 w = fp4_pair(bq); \
                                     run += {av} * w.x + {a1} * w.y; }}"
                                ));
                                self.line("    s += run * (sg * 16384.0f);");
                            }
                            Pack::Pairs(qe, qc) => {
                                // One byte, two values: low nibble for p,
                                // high for p + 1.
                                let byte = at_index(qe, &format!("j * {qc}u + p0 / 2u + u / 2u"));
                                let a1 = Self::read(&ops[0].0, &format!("i * {kd}u + p + 1u"));
                                self.line(&format!(
                                    "    for (uint u = 0; u < {vec}u; u += 2u) {{ const uint p = p0 + u; \
                                     const uint bq = (uint)(uchar)({byte}); \
                                     s += {av} * (float(bq & 0xFu) * sg{mg}) \
                                     + {a1} * (float(bq >> 4u) * sg{mg}); }}"
                                ));
                            }
                            Pack::None => self.line(&format!(
                                "    for (uint u = 0; u < {vec}u; ++u) {{ const uint p = p0 + u; \
                                 s += {av} * ({vv} * sg{mg}); }}"
                            )),
                        }
                        self.line("}");
                    } else if vec > 1 {
                        self.line(&format!(
                            "for (uint p0 = lane * {vec}u; p0 < {kd}u; p0 += {}u) {{",
                            lanes * vec
                        ));
                        self.line(&format!(
                            "    for (uint u = 0; u < {vec}u; ++u) {{ const uint p = p0 + u; s += {av} * {bv}; }}"
                        ));
                        self.line("}");
                    } else {
                        self.line(&format!(
                            "for (uint p = lane; p < {kd}u; p += {lanes}u) s += {av} * {bv};"
                        ));
                    }
                    self.depth -= 1;
                    self.line("}");
                    self.line(&format!(
                        "for (uint d = {}u; d > 0; d /= 2) s += {};",
                        lanes / 2,
                        self.d.shuffle_down("s", "d")
                    ));
                    self.line(&format!(
                        "if (o < {outs}u && lane == 0) scratch[{res} + o] = s;"
                    ));
                    self.depth -= 1;
                    self.line("}");
                    self.barrier();
                    let name = self.declare_reg(x, &reg(*acc, &[m, n]));
                    self.owned(
                        outs,
                        &[format!(
                            "{name}[k] = {};",
                            self.d.convert(*acc, &format!("scratch[{res} + e]"))
                        )],
                    );
                } else {
                    let name = self.declare_reg(x, &reg(*acc, &[m, n]));
                    self.owned(
                        outs,
                        &[
                            format!("const uint i = e / {n}u, j = e % {n}u;"),
                            "float s = 0.0f;".into(),
                            format!("for (uint p = 0; p < {kd}u; ++p) s += {av} * {bv};"),
                            format!("{name}[k] = {};", self.d.convert(*acc, "s")),
                        ],
                    );
                }
            }
            Op::RowReduce(r, a) => {
                let x = dst.ok_or("reduce without a result")?;
                let ta = self.arg_ty(*a)?;
                let (m, n) = (ta.shape[0], ta.shape[1]);
                let ops = self.operands(&[*a], &[false], m)?;
                let (init, comb) = match r {
                    Reduce::Max => ("(-INFINITY)", "max"),
                    Reduce::Sum => ("0.0f", "sum"),
                };
                let join = |a: &str, b: &str| {
                    if comb == "max" {
                        format!("max({a}, {b})")
                    } else {
                        format!("{a} + {b}")
                    }
                };
                // Fewer rows than threads: a group of lanes per row, combined
                // with shuffles within a simdgroup and through scratch across
                // simdgroups. One thread walking a 4096-wide row serially is
                // what made RMSNorm slow.
                let mut lanes = 1;
                while lanes * 2 <= self.threads / m && lanes * 2 <= n {
                    lanes *= 2;
                }
                if lanes >= 2 {
                    let dt = ta.dtype;
                    let sgs = lanes.div_ceil(32);
                    let res = self.staged;
                    self.scratch = self.scratch.max(res + m * sgs);
                    if self.staged == 0 {
                        self.barrier();
                    }
                    let val = Self::read(&ops[0].0, &format!("o * {n}u + j"));
                    self.line("{");
                    self.depth += 1;
                    self.line(&format!(
                        "const uint o = tid / {lanes}u, lane = tid % {lanes}u;"
                    ));
                    self.line(&format!("float s = {init};"));
                    self.line(&format!(
                        "if (o < {m}u) for (uint j = lane; j < {n}u; j += {lanes}u) s = {};",
                        join("s", &val)
                    ));
                    self.line(&format!(
                        "for (uint d = {}u; d > 0; d /= 2) s = {};",
                        lanes.min(32) / 2,
                        join("s", &self.d.shuffle_down("s", "d"))
                    ));
                    self.line(&format!(
                        "if (o < {m}u && lane % 32u == 0) scratch[{res} + o * {sgs}u + lane / 32u] = s;"
                    ));
                    self.depth -= 1;
                    self.line("}");
                    self.barrier();
                    let name = self.declare_reg(x, &reg(dt, &[m]));
                    self.owned(
                        m,
                        &[
                            format!("float s = {init};"),
                            format!(
                                "for (uint q = 0; q < {sgs}u; ++q) s = {};",
                                join("s", &format!("scratch[{res} + e * {sgs}u + q]"))
                            ),
                            format!("{name}[k] = {};", self.d.convert(dt, "s")),
                        ],
                    );
                    return Ok(());
                }
                let val = Self::read(&ops[0].0, &format!("e * {n}u + j"));
                let step = format!("s = {};", join("s", &val));
                let dt = ta.dtype;
                let name = self.declare_reg(x, &reg(dt, &[m]));
                self.owned(
                    m,
                    &[
                        format!("float s = {init};"),
                        format!("for (uint j = 0; j < {n}u; ++j) {step}"),
                        format!("{name}[k] = {};", self.d.convert(dt, "s")),
                    ],
                );
            }
            Op::MakeArray(vs) => {
                let x = dst.ok_or("array without a result")?;
                let mut elems = vec![];
                for &e in vs {
                    let (n, t) = match self.loc(e)? {
                        Loc::Reg(n, t) => (n, t),
                        l @ Loc::Lazy(_, _) => {
                            let Loc::Lazy(_, t) = &l else { unreachable!() };
                            let t = t.clone();
                            let n = format!("{}_m", v(e));
                            let per = self.per(t.elems());
                            self.line(&format!("{} {n}[{per}];", self.d.scalar(t.dtype)));
                            self.assign(&Loc::Reg(n.clone(), t.clone()), &l)?;
                            (n, t)
                        }
                        _ => return Err("arrays of threadgroup tiles are not lowered".into()),
                    };
                    elems.push((n, t));
                }
                let t = elems[0].1.clone();
                let name = v(x);
                let per = self.per(t.elems());
                self.line(&format!(
                    "{} {name}[{}][{per}];",
                    self.d.scalar(t.dtype),
                    elems.len()
                ));
                for (i, (n, _)) in elems.iter().enumerate() {
                    self.line(&format!(
                        "for (uint k = 0; k < {per}u; ++k) {name}[{i}][k] = {n}[k];"
                    ));
                }
                self.locs.insert(x, Loc::RegArr(name, t, vs.len()));
            }
            Op::Acquire(_) | Op::Commit(..) | Op::Receive(_) | Op::Release(..) => {
                return Err("pipes have no Metal lowering (no split barriers)".into());
            }
        }
        Ok(())
    }

    /// `out[i, j] = sum_p a[i, p] * b[j, p]` for a few rows `i` (a small
    /// batch of tokens). Each simdgroup owns `r` weight rows and keeps an
    /// accumulator per (weight row, token); per step it loads each token's
    /// activation once and each weight once, and multiplies every pair. Lazy
    /// dequantised weights get their group terms hoisted and their packed
    /// bytes read once, as in the single-row reduction.
    fn batched_rows(
        &mut self,
        x: Var,
        acc: DType,
        (m, n, kd): (usize, usize, usize),
        b: &Arg,
        ops: &[(Access, TileTy)],
    ) -> Result<(), String> {
        let simd = 32;
        let groups = (self.threads / simd).max(1);
        // Weight rows per simdgroup; the rest of the lowering assumes they
        // tile the output evenly.
        let r = n.div_ceil(groups);
        let lanes = simd;
        let dq = match b {
            Arg::Move(v) | Arg::Borrow(v) => self.dq.get(v).cloned(),
            _ => None,
        };
        let packed = dq.as_ref().is_some_and(|d| !matches!(d.pack, Pack::None));
        let vec = split_k_vec(kd, lanes, if packed { 16 } else { 8 });
        let dq = dq.filter(|d| {
            vec > 1
                && d.cols == kd
                && d.group.is_multiple_of(vec)
                && match d.pack {
                    Pack::None => true,
                    Pack::Pairs(..) | Pack::Fp4(..) => vec.is_multiple_of(2),
                    Pack::Six(..) => vec.is_multiple_of(4),
                }
        });
        // Per step: loads for row `j`, and the weights for p + k.
        let (step, pre, weights): (usize, String, Vec<String>) = match &dq {
            Some(d) => {
                let mg = if d.m.is_some() { " - mgr[rr]" } else { "" };
                match &d.pack {
                    Pack::Pairs(qe, qc) => {
                        let byte = at_index(qe, &format!("j * {qc}u + p0 / 2u + u / 2u"));
                        (
                            2,
                            format!("const uint bq = (uint)(uchar)({byte}); "),
                            vec![
                                format!("(float(bq & 0xFu) * sgr[rr]{mg})"),
                                format!("(float(bq >> 4u) * sgr[rr]{mg})"),
                            ],
                        )
                    }
                    Pack::Fp4(qe, qc) => {
                        let byte = at_index(qe, &format!("j * {qc}u + p0 / 2u + u / 2u"));
                        (
                            2,
                            format!("const float2 bq = fp4_pair((uint)(uchar)({byte})); "),
                            vec![
                                "(bq.x * sgr[rr])".to_string(),
                                "(bq.y * sgr[rr])".to_string(),
                            ],
                        )
                    }
                    Pack::Six(lo, lc, hi, hc) => {
                        let b0 = at_index(lo, &format!("j * {lc}u + p0 / 2u + u / 2u"));
                        let b1 = at_index(lo, &format!("j * {lc}u + p0 / 2u + u / 2u + 1u"));
                        let h = at_index(hi, &format!("j * {hc}u + p0 / 4u + u / 4u"));
                        let q = |l: &str, sh: u32, hs: u32| {
                            format!(
                                "(float(int((({l} >> {sh}u) & 0xFu) | (((hh >> {hs}u) & 3u) << 4u)) - 32) * sgr[rr])"
                            )
                        };
                        (
                            4,
                            format!(
                                "const uint l0 = (uint)(uchar)({b0}), l1 = (uint)(uchar)({b1}), \
                                 hh = (uint)(uchar)({h}); "
                            ),
                            vec![q("l0", 0, 0), q("l0", 4, 2), q("l1", 0, 4), q("l1", 4, 6)],
                        )
                    }
                    Pack::None => {
                        let vv = at_index(&d.v, &format!("j * {kd}u + p"));
                        (1, String::new(), vec![format!("({vv} * sgr[rr]{mg})")])
                    }
                }
            }
            None => (
                1,
                String::new(),
                vec![Self::read(&ops[1].0, &format!("j * {kd}u + p"))],
            ),
        };
        let res = self.staged;
        self.scratch = self.scratch.max(res + m * n);
        if self.staged == 0 {
            self.barrier();
        }
        let v = vec.max(1);
        self.line("{");
        self.depth += 1;
        self.line(&format!(
            "const uint sgid = tid / {simd}u, lane = tid % {simd}u;"
        ));
        // `xa`, `s`, `sgr` and `mgr` are register tiles. Metal keeps them in
        // registers only while every index is a compile-time constant:
        // indexed by a loop variable they go to the stack instead, and a
        // spilled accumulator costs more than the weight reuse it exists for.
        // One loop-variable index anywhere is enough to spill the whole tile,
        // so every touch below is emitted with a literal. `r`, `m` and `step`
        // are all known here.
        let unroll = r * m * step <= 128;
        self.line(&format!("float s[{r}][{m}];"));
        if unroll {
            for rr in 0..r {
                self.line(
                    &(0..m)
                        .map(|i| format!("s[{rr}][{i}] = 0.0f;"))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
        } else {
            self.line(&format!(
                "for (uint rr = 0; rr < {r}u; ++rr) for (uint i = 0; i < {m}u; ++i) s[rr][i] = 0.0f;"
            ));
        }
        self.line(&format!(
            "for (uint p0 = lane * {v}u; p0 < {kd}u; p0 += {}u) {{",
            lanes * v
        ));
        self.depth += 1;
        if let Some(d) = &dq {
            let g = d.group;
            self.line(&format!("float sgr[{r}], mgr[{r}];"));
            let rs = match &d.row {
                // Row-only factor (NVFP4's per-tensor scale).
                Some(r) => format!(" * float({})", at_index(r, "j")),
                None => String::new(),
            };
            // `fp4_pair` leaves its values 2^14 small, so the 2^14 rides
            // on the group's scale: read once a group, not once a value.
            let rs = match d.pack {
                Pack::Fp4(..) => format!("{rs} * 16384.0f"),
                _ => rs,
            };
            let row = |rr: &str, idx: &str| {
                let mg = match &d.m {
                    Some(mm) => format!("mgr[{idx}] = {};", at_index(mm, "grp")),
                    None => format!("mgr[{idx}] = 0.0f;"),
                };
                format!(
                    "{{ const uint j = min(sgid * {r}u + {rr}, {}u); \
                     const uint grp = j * {}u + p0 / {g}u; \
                     sgr[{idx}] = {}{rs}; {mg} }}",
                    n - 1,
                    kd / g,
                    at_index(&d.s, "grp"),
                )
            };
            if unroll {
                for rr in 0..r {
                    self.line(&row(&format!("{rr}u"), &rr.to_string()));
                }
            } else {
                self.line(&format!(
                    "for (uint rr = 0; rr < {r}u; ++rr) {}",
                    row("rr", "rr")
                ));
            }
        }
        self.line(&format!("for (uint u = 0; u < {v}u; u += {step}u) {{"));
        self.depth += 1;
        self.line("const uint p = p0 + u;");
        let names: Vec<String> = (0..weights.len()).map(|k| format!("w{k}")).collect();
        let decl: Vec<String> = weights
            .iter()
            .zip(&names)
            .map(|(w, nm)| format!("const float {nm} = {w};"))
            .collect();
        // Each token's activations for this step, loaded once and reused by
        // every weight row below: that reuse is the whole point of a batch.
        self.line(&format!("float xa[{m}][{step}];"));
        for k in 0..step {
            if unroll {
                for i in 0..m {
                    self.line(&format!(
                        "xa[{i}][{k}] = {};",
                        Self::read(&ops[0].0, &format!("p + {}u", i * kd + k))
                    ));
                }
            } else {
                self.line(&format!(
                    "for (uint i = 0; i < {m}u; ++i) xa[i][{k}] = {};",
                    Self::read(&ops[0].0, &format!("i * {kd}u + p + {k}u"))
                ));
            }
        }
        let terms = |i: &str| -> String {
            names
                .iter()
                .enumerate()
                .map(|(k, nm)| format!("xa[{i}][{k}] * {nm}"))
                .collect::<Vec<_>>()
                .join(" + ")
        };
        if unroll {
            for rr in 0..r {
                // Braces per row: `pre` declares names of its own.
                self.line("{");
                self.depth += 1;
                self.line(&format!(
                    "const uint j = min(sgid * {r}u + {rr}u, {}u);",
                    n - 1
                ));
                // `sgr`/`mgr` are register tiles too, so their row index has
                // to be a literal for the same reason.
                let lit = |s: &str| s.replace("[rr]", &format!("[{rr}]"));
                self.line(&format!(
                    "{}{}",
                    lit(&pre),
                    decl.iter().map(|d| lit(d)).collect::<Vec<_>>().join(" ")
                ));
                for i in 0..m {
                    self.line(&format!("s[{rr}][{i}] += {};", terms(&i.to_string())));
                }
                self.depth -= 1;
                self.line("}");
            }
        } else {
            self.line(&format!("for (uint rr = 0; rr < {r}u; ++rr) {{"));
            self.depth += 1;
            self.line(&format!(
                "const uint j = min(sgid * {r}u + rr, {}u);",
                n - 1
            ));
            self.line(&format!("{pre}{}", decl.join(" ")));
            self.line(&format!(
                "for (uint i = 0; i < {m}u; ++i) s[rr][i] += {};",
                terms("i")
            ));
            self.depth -= 1;
            self.line("}");
        }
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        // The reduction and the store index `s` too, and one loop-variable
        // index anywhere is enough to put the whole tile on the stack.
        if unroll {
            for rr in 0..r {
                for i in 0..m {
                    self.line(&format!(
                        "for (uint d = {}u; d > 0; d /= 2) \
                         s[{rr}][{i}] += {};",
                        lanes / 2,
                        self.d.shuffle_down(&format!("s[{rr}][{i}]"), "d")
                    ));
                }
            }
            self.line("if (lane == 0) {");
            self.depth += 1;
            for rr in 0..r {
                self.line(&format!(
                    "{{ const uint j = sgid * {r}u + {rr}u; if (j < {n}u) {{ {} }} }}",
                    (0..m)
                        .map(|i| format!("scratch[{res} + {}u + j] = s[{rr}][{i}];", i * n))
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
            }
            self.depth -= 1;
            self.line("}");
        } else {
            self.line(&format!(
                "for (uint rr = 0; rr < {r}u; ++rr) for (uint i = 0; i < {m}u; ++i) \
                 for (uint d = {}u; d > 0; d /= 2) s[rr][i] += {};",
                lanes / 2,
                self.d.shuffle_down("s[rr][i]", "d")
            ));
            self.line(&format!(
                "if (lane == 0) for (uint rr = 0; rr < {r}u; ++rr) {{ const uint j = sgid * {r}u + rr; \
                 if (j < {n}u) for (uint i = 0; i < {m}u; ++i) scratch[{res} + i * {n}u + j] = s[rr][i]; }}"
            ));
        }
        self.depth -= 1;
        self.line("}");
        self.barrier();
        let name = self.declare_reg(x, &TileTy::new(acc, &[m, n], Space::Reg));
        self.owned(
            m * n,
            &[format!(
                "{name}[k] = {};",
                self.d.convert(acc, &format!("scratch[{res} + e]"))
            )],
        );
        Ok(())
    }

    /// The expression and type of a lazy operand, if it is one.
    fn lazy(&self, a: Arg) -> Option<(String, TileTy)> {
        match a {
            Arg::Move(x) | Arg::Borrow(x) => match self.locs.get(&x) {
                Some(Loc::Lazy(e, t)) => Some((e.clone(), t.clone())),
                _ => None,
            },
            _ => None,
        }
    }

    fn arg_ty(&self, a: Arg) -> Result<TileTy, String> {
        Ok(match a {
            Arg::Move(x) | Arg::Borrow(x) => match self.loc(x)? {
                Loc::Reg(_, t) | Loc::Tg(_, t) | Loc::Lazy(_, t) => t,
                _ => return Err("operand is not a tile".into()),
            },
            Arg::BorrowElem(arr, _) => match self.loc(arr)? {
                Loc::RegArr(_, t, _) => t,
                _ => return Err("operand is not an array".into()),
            },
            Arg::BorrowPart(..) => return Err("pipe shares have no Metal lowering".into()),
        })
    }
}
