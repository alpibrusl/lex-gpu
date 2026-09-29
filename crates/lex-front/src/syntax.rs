//! A surface for `.lx`: text in, the same typed IR out.
//!
//! Until now a program was Rust that called [`crate::ir::Builder`], and
//! `docs/design.md` kept the syntax unbuilt on an explicit condition —
//! that the parts worth arguing about "are exactly the parts a second
//! target would rewrite", so the surface should wait for a second target.
//! There are two backends now, and they disagree about real things, so
//! the condition is met and this is the first slice of the answer.
//!
//! What it does today: `algo` declarations with parameters, `let`
//! bindings over the elementwise and reduction ops, a `store`, and
//! `schedule` blocks that set `threads` and `chunk` per target. It parses
//! `rmsnorm` and `silu_mul` into programs byte-identical to the Rust ones,
//! which is the whole of its claim — see `tests/syntax.rs`, which holds it
//! to the emitter goldens rather than to itself.
//!
//! What it does not do, stated because a surface that hides its holes is
//! worse than no surface: no loops, no index arithmetic, no layouts or
//! memory spaces in the type, no autotuner `?`. Those are the interesting
//! half and they are next; this half is the part that has to exist before
//! any of them can be written down.
//!
//! ## The linear discipline is visible
//!
//! A tile is consumed exactly once. The IR says so with `Arg::Move` and
//! `Arg::Borrow`, and the surface says it with a sigil:
//!
//! ```text
//! let sq = &x * &x     // borrowed twice, x lives on
//! let ss = rowsum sq   // moved, sq is gone
//! ```
//!
//! Making that a sigil rather than an inference is deliberate. The
//! checker's whole value is that using a consumed tile is a compile
//! error; a surface that guessed which uses were moves would move the
//! error from the reader's eye to the compiler's discretion.

use std::collections::HashMap;

use crate::ir::{Arg, BinOp, Builder, IdxExpr, Op, Program, Reduce, TileTy, UnOp, View};
use lex_ir::{DType, Space};

/// Where a token starts, for errors that can be acted on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct At {
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for At {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    /// One of `( ) { } [ ] , : = * + - / & ->`
    Punct(String),
    End,
}

struct Lexer<'a> {
    src: std::iter::Peekable<std::str::Chars<'a>>,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src: src.chars().peekable(),
            line: 1,
            col: 1,
        }
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.src.next()?;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn tokens(mut self) -> Result<Vec<(Tok, At)>, String> {
        let mut out = vec![];
        loop {
            // Whitespace and `//` comments.
            while let Some(&c) = self.src.peek() {
                if c.is_whitespace() {
                    self.bump();
                } else if c == '/' {
                    // Only a comment if doubled; a lone `/` divides.
                    let at = At {
                        line: self.line,
                        col: self.col,
                    };
                    self.bump();
                    if self.src.peek() == Some(&'/') {
                        while let Some(c) = self.bump() {
                            if c == '\n' {
                                break;
                            }
                        }
                    } else {
                        out.push((Tok::Punct("/".into()), at));
                    }
                } else {
                    break;
                }
            }
            let at = At {
                line: self.line,
                col: self.col,
            };
            let Some(&c) = self.src.peek() else {
                out.push((Tok::End, at));
                return Ok(out);
            };
            if c.is_alphabetic() || c == '_' {
                let mut s = String::new();
                while let Some(&c) = self.src.peek() {
                    if c.is_alphanumeric() || c == '_' {
                        s.push(c);
                        self.bump();
                    } else {
                        break;
                    }
                }
                out.push((Tok::Ident(s), at));
            } else if c.is_ascii_digit() {
                let mut s = String::new();
                while let Some(&c) = self.src.peek() {
                    // `1e-5` and `0.5` both have to lex as one number, so
                    // a sign is part of it only straight after an
                    // exponent marker -- otherwise `a-1` would lex as two
                    // tokens on one side and a negative literal on the
                    // other depending on the spacing.
                    let exponent_sign =
                        (c == '-' || c == '+') && matches!(s.chars().last(), Some('e' | 'E'));
                    if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || exponent_sign {
                        s.push(c);
                        self.bump();
                    } else {
                        break;
                    }
                }
                let n: f64 = s
                    .parse()
                    .map_err(|_| format!("{at}: `{s}` is not a number"))?;
                out.push((Tok::Num(n), at));
            } else {
                self.bump();
                // `->` is one token; everything else is one character.
                if c == '-' && self.src.peek() == Some(&'>') {
                    self.bump();
                    out.push((Tok::Punct("->".into()), at));
                } else if "(){}[],:=*+-/&".contains(c) {
                    out.push((Tok::Punct(c.to_string()), at));
                } else {
                    return Err(format!("{at}: `{c}` means nothing here"));
                }
            }
        }
    }
}

/// A parameter: `in x: f32[1, n]` or `out y: f32[1, n]`.
struct ParamDecl {
    name: String,
    dtype: DType,
    shape: Vec<Dim>,
    writable: bool,
}

/// A dimension is a literal or one of the algorithm's constants.
#[derive(Clone, Debug)]
enum Dim {
    Lit(usize),
    Named(String),
}

/// The right-hand side of a `let`, or the value of a `store`.
#[derive(Clone, Debug)]
enum Expr {
    /// `x`, moved.
    Move(String),
    /// `&x`, borrowed.
    Borrow(String),
    /// A constant, which becomes a `Fill` of the shape it meets.
    Num(f64),
    /// `load x`
    Load(String),
    /// `rowsum e`, `rsqrt e`, ...
    Call(String, Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
}

struct Parser {
    toks: Vec<(Tok, At)>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.i].0
    }

    fn at(&self) -> At {
        self.toks[self.i].1
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.i].0.clone();
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        t
    }

    fn eat_punct(&mut self, p: &str) -> bool {
        if matches!(self.peek(), Tok::Punct(s) if s == p) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn want_punct(&mut self, p: &str) -> Result<(), String> {
        if self.eat_punct(p) {
            Ok(())
        } else {
            Err(format!(
                "{}: expected `{p}`, found {:?}",
                self.at(),
                self.peek()
            ))
        }
    }

    fn want_ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("{}: expected a name, found {other:?}", self.at())),
        }
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if matches!(self.peek(), Tok::Ident(s) if s == kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn dim(&mut self) -> Result<Dim, String> {
        match self.bump() {
            Tok::Num(n) => Ok(Dim::Lit(n as usize)),
            Tok::Ident(s) => Ok(Dim::Named(s)),
            other => Err(format!(
                "{}: expected a dimension, found {other:?}",
                self.at()
            )),
        }
    }

    /// `f32[1, n]`
    fn ty(&mut self) -> Result<(DType, Vec<Dim>), String> {
        let at = self.at();
        let name = self.want_ident()?;
        let dtype = match name.as_str() {
            "f32" => DType::F32,
            "f16" => DType::F16,
            "i8" => DType::I8,
            other => return Err(format!("{at}: `{other}` is not a dtype this slice knows")),
        };
        self.want_punct("[")?;
        let mut shape = vec![self.dim()?];
        while self.eat_punct(",") {
            shape.push(self.dim()?);
        }
        self.want_punct("]")?;
        Ok((dtype, shape))
    }

    /// Unary and atoms; `*` and `/` bind tighter than `+` and `-`.
    fn atom(&mut self) -> Result<Expr, String> {
        let at = self.at();
        if self.eat_punct("&") {
            return Ok(Expr::Borrow(self.want_ident()?));
        }
        if self.eat_punct("(") {
            let e = self.expr()?;
            self.want_punct(")")?;
            return Ok(e);
        }
        match self.bump() {
            Tok::Num(n) => Ok(Expr::Num(n)),
            Tok::Ident(s) => match s.as_str() {
                "load" => Ok(Expr::Load(self.want_ident()?)),
                // A call is `name expr` with no parentheses, which reads
                // as the pipeline it is: `rowsum sq`, `rsqrt t`.
                "rowsum" | "rowmax" | "rsqrt" | "sigmoid" | "softplus" => {
                    Ok(Expr::Call(s, Box::new(self.atom()?)))
                }
                _ => Ok(Expr::Move(s)),
            },
            other => Err(format!("{at}: expected a value, found {other:?}")),
        }
    }

    fn product(&mut self) -> Result<Expr, String> {
        let mut lhs = self.atom()?;
        loop {
            let op = if self.eat_punct("*") {
                BinOp::Mul
            } else if self.eat_punct("/") {
                BinOp::Div
            } else {
                return Ok(lhs);
            };
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(self.atom()?));
        }
    }

    fn expr(&mut self) -> Result<Expr, String> {
        let mut lhs = self.product()?;
        loop {
            let op = if self.eat_punct("+") {
                BinOp::Add
            } else if self.eat_punct("-") {
                BinOp::Sub
            } else {
                return Ok(lhs);
            };
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(self.product()?));
        }
    }
}

/// One statement of an `algo` body.
enum Stmt {
    Let(String, Expr),
    Store(Expr, String),
}

/// A parsed algorithm, before any target is chosen.
pub struct Algo {
    pub name: String,
    consts: Vec<String>,
    params: Vec<ParamDecl>,
    body: Vec<Stmt>,
}

/// What a target binds that the algorithm leaves open.
///
/// The algorithm says what to compute and contains no target anywhere in
/// it; the schedule says how, for one machine. That split is the design's
/// central claim and the reason the surface waited for a second backend:
/// with one target there is nothing to split, and any syntax invented for
/// it would be a syntax invented twice.
///
/// What a schedule can set today is `threads`. The design has tile sizes,
/// MMA atoms, copy staging and layouts here too, and the measurement that
/// most wants to live here -- rows per simdgroup, 1 on Apple and 2 on Ada
/// -- is still a constant in the target table because no rule fits both
/// of the shapes measured. That is the next thing this grows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// The `Target::name` this binds to, e.g. `apple-m-series`.
    pub target: String,
    pub threads: usize,
    /// Elements one grid instance owns, partitioning the trailing
    /// dimension. `None` runs the whole thing in one instance.
    ///
    /// This is here and not in the algorithm on purpose, and it is the
    /// clearest case for the split in the language so far. What the
    /// kernel means is `y = silu(g) * u` over n elements; cutting n into
    /// pieces of 256 is a decision about a machine. The algorithm does
    /// not mention it, the schedule does, and the same algorithm takes a
    /// different cut on a different card without being rewritten.
    pub chunk: Option<usize>,
}

/// One `.lx` file: an algorithm, and the schedules that bind it.
pub struct Unit {
    pub algo: Algo,
    pub schedules: Vec<Schedule>,
}

impl Unit {
    /// The schedule for a target, by its `Target::name`.
    ///
    /// An error rather than a default: a program with no schedule for the
    /// machine it is being compiled for is a program nobody has chosen a
    /// shape for, and quietly picking one is how a 3x slowdown becomes
    /// somebody's afternoon.
    pub fn schedule_for(&self, target: &str) -> Result<&Schedule, String> {
        self.schedules
            .iter()
            .find(|s| s.target == target)
            .ok_or_else(|| {
                let have: Vec<&str> = self.schedules.iter().map(|s| s.target.as_str()).collect();
                format!(
                    "`{}` has no schedule for `{target}`; it has {have:?}",
                    self.algo.name
                )
            })
    }
}

impl Unit {
    /// The algorithm, bound to one target's schedule.
    ///
    /// This is the whole split in one call: the constants come from the
    /// caller, the machine-shaped decisions come from the file's
    /// `schedule` block, and what comes back is a program plus the thread
    /// count to lower it with.
    pub fn compile(&self, target: &str, args: &[(&str, f64)]) -> Result<(Program, usize), String> {
        let s = self.schedule_for(target)?;
        Ok((self.algo.build_with(args, s.chunk)?, s.threads))
    }
}

/// Parse one `algo` from `.lx` source.
pub fn parse(src: &str) -> Result<Unit, String> {
    let toks = Lexer::new(src).tokens()?;
    let mut p = Parser { toks, i: 0 };

    if !p.eat_kw("algo") {
        return Err(format!("{}: a file starts with `algo`", p.at()));
    }
    let name = p.want_ident()?;

    // `algo rmsnorm(n, eps)`: the constants a caller supplies. They are
    // shapes and scalars, not tiles, so they carry no type here.
    let mut consts = vec![];
    if p.eat_punct("(") {
        while !p.eat_punct(")") {
            consts.push(p.want_ident()?);
            if !p.eat_punct(",") {
                p.want_punct(")")?;
                break;
            }
        }
    }

    let mut params = vec![];
    loop {
        let writable = if p.eat_kw("in") {
            false
        } else if p.eat_kw("out") {
            true
        } else {
            break;
        };
        let name = p.want_ident()?;
        p.want_punct(":")?;
        let (dtype, shape) = p.ty()?;
        params.push(ParamDecl {
            name,
            dtype,
            shape,
            writable,
        });
    }
    if params.is_empty() {
        return Err(format!("{}: an algo needs at least one parameter", p.at()));
    }

    p.want_punct("{")?;
    let mut body = vec![];
    while !p.eat_punct("}") {
        if p.eat_kw("let") {
            let dst = p.want_ident()?;
            p.want_punct("=")?;
            body.push(Stmt::Let(dst, p.expr()?));
        } else if p.eat_kw("store") {
            let e = p.expr()?;
            p.want_punct("->")?;
            body.push(Stmt::Store(e, p.want_ident()?));
        } else {
            return Err(format!(
                "{}: expected `let`, `store` or `}}`, found {:?}",
                p.at(),
                p.peek()
            ));
        }
    }
    let algo = Algo {
        name,
        consts,
        params,
        body,
    };

    // `schedule <algo> for <target> { threads N }`
    let mut schedules = vec![];
    while p.eat_kw("schedule") {
        let at = p.at();
        let for_algo = p.want_ident()?;
        if for_algo != algo.name {
            return Err(format!(
                "{at}: this file declares `{}`, not `{for_algo}`",
                algo.name
            ));
        }
        if !p.eat_kw("for") {
            return Err(format!("{}: expected `for <target>`", p.at()));
        }
        // Target names carry dashes, which do not lex as one identifier.
        let mut target = p.want_ident()?;
        while p.eat_punct("-") {
            target.push('-');
            target.push_str(&p.want_ident()?);
        }
        p.want_punct("{")?;
        let (mut threads, mut chunk) = (None, None);
        while !p.eat_punct("}") {
            let at = p.at();
            let key = p.want_ident()?;
            match key.as_str() {
                "threads" => match p.bump() {
                    Tok::Num(n) => threads = Some(n as usize),
                    other => return Err(format!("{at}: threads takes a number, found {other:?}")),
                },
                "chunk" => match p.bump() {
                    Tok::Num(n) => chunk = Some(n as usize),
                    other => return Err(format!("{at}: chunk takes a number, found {other:?}")),
                },
                other => return Err(format!("{at}: `{other}` is not a schedule key")),
            }
            // A comma between keys is allowed and means nothing. The
            // design doc writes them on separate lines without; a writer
            // who reaches for one anyway should not get a parse error
            // about a name.
            p.eat_punct(",");
        }
        let threads =
            threads.ok_or_else(|| format!("{at}: the schedule for `{target}` sets no threads"))?;
        if schedules.iter().any(|s: &Schedule| s.target == target) {
            return Err(format!("{at}: two schedules for `{target}`"));
        }
        schedules.push(Schedule {
            target,
            threads,
            chunk,
        });
    }

    if !matches!(p.peek(), Tok::End) {
        return Err(format!("{}: trailing input after the algo", p.at()));
    }
    Ok(Unit { algo, schedules })
}

impl Algo {
    /// Build the IR, binding the algorithm's constants.
    ///
    /// `name` is formed as the Rust builders form it -- `rmsnorm_4096` --
    /// because the emitter's golden files are keyed by it and this has to
    /// produce the same program, not merely an equivalent one.
    pub fn build(&self, args: &[(&str, f64)]) -> Result<Program, String> {
        self.build_with(args, None)
    }

    /// [`Algo::build`], partitioning the trailing dimension into pieces of
    /// `chunk`, one grid instance each.
    ///
    /// Every parameter is cut the same way and every load becomes this
    /// instance's slice, which is why the algorithm can be written as if
    /// it saw the whole tensor. The restriction is that they must all
    /// have the same trailing dimension: a kernel whose inputs are cut
    /// differently is expressing something this cannot say, and saying so
    /// is better than cutting one of them wrongly.
    pub fn build_with(
        &self,
        args: &[(&str, f64)],
        chunk: Option<usize>,
    ) -> Result<Program, String> {
        let vals: HashMap<&str, f64> = args.iter().copied().collect();
        for c in &self.consts {
            if !vals.contains_key(c.as_str()) {
                return Err(format!("`{}` needs a value for `{c}`", self.name));
            }
        }
        let dim = |d: &Dim| -> Result<usize, String> {
            Ok(match d {
                Dim::Lit(n) => *n,
                Dim::Named(s) => *vals
                    .get(s.as_str())
                    .ok_or_else(|| format!("`{s}` is not one of this algo's constants"))?
                    as usize,
            })
        };

        // The suffix the Rust builders use: every constant in declaration
        // order. `rmsnorm(n)` with n = 4096 is `rmsnorm_4096`.
        let mut full = self.name.clone();
        for c in &self.consts {
            let v = vals[c.as_str()];
            if v.fract() == 0.0 && v >= 0.0 {
                full.push_str(&format!("_{}", v as usize));
            }
        }
        let mut b = Builder::new(&full);

        // A constant used in an expression is a number, not a tile. `eps`
        // reads like a value because it is one; it becomes a `Fill` of
        // whatever shape it is added to, which is decided where it lands
        // rather than where it is written.
        let subst = |e: &Expr| -> Expr { substitute(e, &vals) };

        // Declare every parameter at its full shape, then decide what one
        // instance sees.
        let mut full_shapes: Vec<Vec<usize>> = vec![];
        for p in &self.params {
            full_shapes.push(p.shape.iter().map(dim).collect::<Result<_, _>>()?);
        }
        let trailing: Vec<usize> = full_shapes
            .iter()
            .map(|s| *s.last().expect("a shape has a dimension"))
            .collect();
        if chunk.is_some() && trailing.windows(2).any(|w| w[0] != w[1]) {
            return Err(format!(
                "`{}`: a chunked algo needs one trailing dimension, not {trailing:?}",
                self.name
            ));
        }
        let pid = match chunk {
            Some(c) => {
                let n = trailing[0];
                if !n.is_multiple_of(c) {
                    return Err(format!("`{}`: chunk {c} does not divide {n}", self.name));
                }
                Some((b.grid(n / c), c))
            }
            None => None,
        };

        let mut params: HashMap<&str, (usize, Vec<usize>, DType)> = HashMap::new();
        for (p, full) in self.params.iter().zip(&full_shapes) {
            let id = b.param(&p.name, p.dtype, full, p.writable);
            // What the body sees: the whole thing, or this instance's cut.
            let seen = match pid {
                Some((_, c)) => {
                    let mut s = full.clone();
                    *s.last_mut().expect("checked above") = c;
                    s
                }
                None => full.clone(),
            };
            params.insert(&p.name, (id, seen, p.dtype));
        }

        // Tiles in scope. A name maps to the IR var and the shape it has,
        // so a `Fill` can take the shape of whatever it is added to.
        let mut env: HashMap<String, (crate::ir::Var, Vec<usize>, DType)> = HashMap::new();

        for stmt in &self.body {
            match stmt {
                Stmt::Let(dst, e) => {
                    let (v, shape, dt) =
                        self.lower(&mut b, &mut env, &params, &subst(e), dst, pid)?;
                    env.insert(dst.clone(), (v, shape, dt));
                }
                Stmt::Store(e, dst) => {
                    let (v, _, _) = self.lower(&mut b, &mut env, &params, &subst(e), "y", pid)?;
                    let (id, shape, _) = params
                        .get(dst.as_str())
                        .ok_or_else(|| format!("`{dst}` is not a parameter"))?;
                    b.effect(Op::Store(Arg::Move(v), slice(pid, *id, shape)));
                }
            }
        }
        Ok(b.finish())
    }

    #[allow(clippy::type_complexity)]
    fn lower(
        &self,
        b: &mut Builder,
        env: &mut HashMap<String, (crate::ir::Var, Vec<usize>, DType)>,
        params: &HashMap<&str, (usize, Vec<usize>, DType)>,
        e: &Expr,
        name: &str,
        pid: Option<(crate::ir::Var, usize)>,
    ) -> Result<(crate::ir::Var, Vec<usize>, DType), String> {
        Ok(match e {
            Expr::Load(p) => {
                let (id, shape, dt) = params
                    .get(p.as_str())
                    .ok_or_else(|| format!("`{p}` is not a parameter"))?;
                let v = b.op(
                    p,
                    Op::Load(slice(pid, *id, shape), TileTy::new(*dt, shape, Space::Reg)),
                );
                (v, shape.clone(), *dt)
            }
            Expr::Move(n) | Expr::Borrow(n) => {
                let (v, shape, dt) = env
                    .get(n)
                    .ok_or_else(|| format!("`{n}` is not bound"))?
                    .clone();
                (v, shape, dt)
            }
            Expr::Num(_) => return Err("a constant needs something to take its shape from".into()),
            Expr::Call(f, inner) => {
                let (v, shape, dt) = self.lower(b, env, params, inner, name, pid)?;
                let a = arg(inner, v);
                match f.as_str() {
                    "rowsum" | "rowmax" => {
                        let r = if f == "rowsum" {
                            Reduce::Sum
                        } else {
                            Reduce::Max
                        };
                        // A row reduction of `[rows, n]` gives `[rows]`,
                        // one dimension shorter -- not `[rows, 1]`. The
                        // checker broadcasts the 1-d form against the 2-d
                        // one, which is how `x * rsqrt(mean(x^2))` works
                        // at all, and the two-dimensional shape would be
                        // rejected as a mismatch.
                        (b.op(name, Op::RowReduce(r, a)), vec![shape[0]], dt)
                    }
                    _ => {
                        // Only the unary ops the IR has. A surface that
                        // offers `exp` because a language usually has one
                        // would be a surface for a different compiler.
                        let u = match f.as_str() {
                            "rsqrt" => UnOp::Rsqrt,
                            "sigmoid" => UnOp::Sigmoid,
                            "softplus" => UnOp::Softplus,
                            other => return Err(format!("`{other}` is not an operation here")),
                        };
                        (b.op(name, Op::Unary(u, a)), shape, dt)
                    }
                }
            }
            Expr::Bin(op, l, r) => {
                // A literal on either side becomes a scale or a fill: a
                // multiply by a constant is `Scale`, which the emitter
                // folds, and anything else needs a tile to add to.
                if let (BinOp::Mul, Expr::Num(k)) = (op, r.as_ref()) {
                    let (v, shape, dt) = self.lower(b, env, params, l, name, pid)?;
                    return Ok((b.op(name, Op::Scale(arg(l, v), *k as f32)), shape, dt));
                }
                if let (BinOp::Mul, Expr::Num(k)) = (op, l.as_ref()) {
                    let (v, shape, dt) = self.lower(b, env, params, r, name, pid)?;
                    return Ok((b.op(name, Op::Scale(arg(r, v), *k as f32)), shape, dt));
                }
                let (lv, lshape, dt) = self.lower(b, env, params, l, name, pid)?;
                let (rv, rshape, _) = match r.as_ref() {
                    Expr::Num(k) => {
                        // Shaped like what it meets, so `ms + eps` fills a
                        // `[rows]` tile and not a `[rows, n]` one.
                        let t = TileTy::new(dt, &lshape, Space::Reg);
                        (b.op("eps", Op::Fill(t, *k as f32)), lshape.clone(), dt)
                    }
                    other => self.lower(b, env, params, other, name, pid)?,
                };
                // The checker's two broadcasts, and no others: a 1-d
                // tile of `rows` against `[rows, n]`, and a single row
                // against every row.
                let row_b = lshape.len() == 2 && rshape == [lshape[0]];
                let col_b = lshape.len() == 2 && rshape == [1, lshape[1]];
                if lshape != rshape && !row_b && !col_b {
                    return Err(format!(
                        "`{name}`: {lshape:?} against {rshape:?}, which do not broadcast"
                    ));
                }
                let (la, ra) = (arg(l, lv), arg_r(r, rv));
                (b.op(name, Op::Binary(*op, la, ra)), lshape, dt)
            }
        })
    }
}

/// Replace a use of an algorithm constant with its value.
///
/// Only a bare use: `&eps` would be borrowing a number, which is not a
/// thing, and leaving it alone makes the error say "not bound" at the
/// name rather than somewhere inside the lowering.
fn substitute(e: &Expr, vals: &HashMap<&str, f64>) -> Expr {
    match e {
        Expr::Move(n) => match vals.get(n.as_str()) {
            Some(v) => Expr::Num(*v),
            None => e.clone(),
        },
        Expr::Call(f, x) => Expr::Call(f.clone(), Box::new(substitute(x, vals))),
        Expr::Bin(op, l, r) => {
            let (l, r) = (substitute(l, vals), substitute(r, vals));
            // Fold once both sides are numbers, so `1 / n` is a constant
            // the multiply can become a `Scale` of, rather than a divide
            // of two filled tiles.
            //
            // In f32, because that is where the answer lands and where
            // the Rust computes it. Dividing in f64 and narrowing rounds
            // twice, which for most `n` gives the same bits and for some
            // gives a different last one -- and "the same program" here
            // means the same constant in the emitted text.
            if let (Expr::Num(a), Expr::Num(b)) = (&l, &r) {
                let (a, b) = (*a as f32, *b as f32);
                let v = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    _ => return Expr::Bin(*op, Box::new(l), Box::new(r)),
                };
                return Expr::Num(v as f64);
            }
            Expr::Bin(*op, Box::new(l), Box::new(r))
        }
        _ => e.clone(),
    }
}

/// `&x` borrows; anything else moves. The sigil is the whole of it, so
/// that a reader can see which uses the checker will count.
fn arg(e: &Expr, v: crate::ir::Var) -> Arg {
    match e {
        Expr::Borrow(_) => Arg::Borrow(v),
        _ => Arg::Move(v),
    }
}

/// The right operand of a binary op, where a literal became a `Fill` and
/// is therefore always moved.
fn arg_r(e: &Expr, v: crate::ir::Var) -> Arg {
    match e {
        Expr::Borrow(_) => Arg::Borrow(v),
        _ => Arg::Move(v),
    }
}

/// What one grid instance sees of a parameter: the whole thing, or its
/// own cut of the trailing dimension.
fn slice(pid: Option<(crate::ir::Var, usize)>, id: usize, shape: &[usize]) -> View {
    match pid {
        Some((pid, c)) => {
            let mut offset: Vec<IdxExpr> = shape.iter().map(|_| IdxExpr::lit(0)).collect();
            if let Some(last) = offset.last_mut() {
                *last = IdxExpr::scaled(pid, c, 0);
            }
            View {
                param: id,
                offset,
                shape: shape.to_vec(),
            }
        }
        None => whole(id, shape),
    }
}

/// A view of a whole parameter.
fn whole(param: usize, shape: &[usize]) -> View {
    View {
        param,
        offset: shape.iter().map(|_| IdxExpr::lit(0)).collect(),
        shape: shape.to_vec(),
    }
}
