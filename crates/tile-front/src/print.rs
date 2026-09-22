//! Text form of a program, for humans. Not parsed back.

use std::fmt::Write;

use crate::ir::{Arg, Block, IdxExpr, Op, PipeTy, Program, Reduce, Stmt, TileTy, Ty, Var, View};

pub fn program(p: &Program) -> String {
    let mut out = String::new();
    let params: Vec<String> = p
        .params
        .iter()
        .map(|q| {
            let rw = if q.writable { "out " } else { "" };
            format!("{rw}{}: {:?}{:?}", q.name, q.dtype, q.shape)
        })
        .collect();
    let _ = writeln!(out, "kernel {}({}) {{", p.name, params.join(", "));
    block(p, &p.body, 1, &mut out);
    out.push_str("}\n");
    out
}

fn tile(t: &TileTy) -> String {
    format!("{:?}{:?}@{:?}", t.dtype, t.shape, t.space)
}

fn ty(t: &Ty) -> String {
    match t {
        Ty::Tile(t) => tile(t),
        Ty::Future(t) => format!("Future<{}>", tile(t)),
        Ty::Array(t, n) => format!("[{}; {n}]", tile(t)),
        Ty::Index => "index".into(),
        Ty::Producer(q) => format!("Producer<{}>", pipe(q)),
        Ty::Consumer(q, k) => format!("Consumer#{k}<{}>", pipe(q)),
        Ty::Slot(q) => format!("Slot<{}>", pipe(q)),
        Ty::Share(q, k) => format!("Share#{k}<{}>", pipe(q)),
    }
}

fn pipe(q: &PipeTy) -> String {
    let parts: Vec<String> = q.elem.iter().map(tile).collect();
    format!(
        "({}) x{}, {} consumers",
        parts.join(", "),
        q.stages,
        q.consumers
    )
}

fn var(p: &Program, v: Var) -> String {
    format!("%{}", p.name_of(v))
}

fn arg(p: &Program, a: Arg) -> String {
    match a {
        Arg::Move(v) => var(p, v),
        Arg::Borrow(v) => format!("&{}", var(p, v)),
        Arg::BorrowElem(a, i) => format!("&{}[{}]", var(p, a), var(p, i)),
        Arg::BorrowPart(s, k) => format!("&{}.{k}", var(p, s)),
    }
}

fn expr(p: &Program, e: &IdxExpr) -> String {
    let mut parts: Vec<String> = e
        .terms
        .iter()
        .map(|&(v, c)| {
            if c == 1 {
                var(p, v)
            } else {
                format!("{c}*{}", var(p, v))
            }
        })
        .collect();
    if e.constant != 0 || parts.is_empty() {
        parts.push(e.constant.to_string());
    }
    parts.join("+")
}

fn view(p: &Program, v: &View) -> String {
    let off: Vec<String> = v.offset.iter().map(|e| expr(p, e)).collect();
    format!(
        "{}[{}]{:?}",
        p.params[v.param].name,
        off.join(", "),
        v.shape
    )
}

fn vars(p: &Program, vs: &[Var]) -> String {
    vs.iter().map(|&v| var(p, v)).collect::<Vec<_>>().join(", ")
}

fn op(p: &Program, o: &Op) -> String {
    match o {
        Op::Alloc(t) => format!("alloc {}", tile(t)),
        Op::Fill(t, x) => format!("fill {} {x}", tile(t)),
        Op::Load(v, t) => format!("load {} -> {}", view(p, v), tile(t)),
        Op::CopyAsync(v, d) => format!("copy_async {} -> {}", view(p, v), var(p, *d)),
        Op::Wait(f) => format!("wait {}", var(p, *f)),
        Op::Store(a, v) => format!("store {} -> {}", arg(p, *a), view(p, v)),
        Op::Drop(v) => format!("drop {}", var(p, *v)),
        Op::Dup(a) => format!("dup {}", arg(p, *a)),
        Op::MatMulNT(a, b, acc) => format!("matmul_nt {}, {} acc {acc:?}", arg(p, *a), arg(p, *b)),
        Op::MatMul(a, b, acc) => format!("matmul {}, {} acc {acc:?}", arg(p, *a), arg(p, *b)),
        Op::Binary(bop, a, b) => format!("{} {}, {}", bop.name(), arg(p, *a), arg(p, *b)),
        Op::Exp(a) => format!("exp {}", arg(p, *a)),
        Op::Scale(a, s) => format!("scale {}, {s}", arg(p, *a)),
        Op::RowReduce(r, a) => {
            let n = match r {
                Reduce::Max => "rowmax",
                Reduce::Sum => "rowsum",
            };
            format!("{n} {}", arg(p, *a))
        }
        Op::Convert(a, dt) => format!("convert {} -> {dt:?}", arg(p, *a)),
        Op::MakeArray(vs) => format!("array [{}]", vars(p, vs)),
        Op::Acquire(h) => format!("acquire {}", var(p, *h)),
        Op::Commit(h, views, slot) => {
            let vs: Vec<String> = views.iter().map(|v| view(p, v)).collect();
            format!(
                "commit {} [{}] -> {}",
                var(p, *h),
                vs.join(", "),
                var(p, *slot)
            )
        }
        Op::Receive(h) => format!("receive {}", var(p, *h)),
        Op::Release(h, s) => format!("release {}, {}", var(p, *h), var(p, *s)),
    }
}

fn block(p: &Program, b: &Block, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    for s in &b.stmts {
        match s {
            Stmt::Let {
                dst: Some(d),
                op: o,
            } => {
                let _ = writeln!(out, "{pad}{} = {}", var(p, *d), op(p, o));
            }
            Stmt::Let { dst: None, op: o } => {
                let _ = writeln!(out, "{pad}{}", op(p, o));
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
                let carry: Vec<String> = params
                    .iter()
                    .zip(init)
                    .map(|(&q, &i)| {
                        let t = p.declared[q.0 as usize].as_ref().map_or("?".into(), ty);
                        format!("{}: {t} = {}", var(p, q), var(p, i))
                    })
                    .collect();
                let head = format!("for {} in {start}..{end}", var(p, *index));
                if carry.is_empty() {
                    let _ = writeln!(out, "{pad}{head} {{");
                } else {
                    let _ = writeln!(out, "{pad}{} = {head} carry(", vars(p, results));
                    for c in carry {
                        let _ = writeln!(out, "{pad}    {c},");
                    }
                    let _ = writeln!(out, "{pad}) {{");
                }
                block(p, body, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
            }
            Stmt::MapEach {
                index,
                arrays,
                params,
                body,
                results,
            } => {
                let lhs = if results.is_empty() {
                    String::new()
                } else {
                    format!("{} = ", vars(p, results))
                };
                let _ = writeln!(
                    out,
                    "{pad}{lhs}map_each {} ({}) in ({}) {{",
                    var(p, *index),
                    vars(p, params),
                    vars(p, arrays)
                );
                block(p, body, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
            }
            Stmt::Specialize { pipes, roles } => {
                let _ = writeln!(out, "{pad}specialize {{");
                for q in pipes {
                    let _ = writeln!(
                        out,
                        "{pad}  pipe {} -> [{}]: {}",
                        var(p, q.ty.id),
                        vars(p, &q.consumers),
                        pipe(&q.ty)
                    );
                }
                for r in roles {
                    let lhs = if r.results.is_empty() {
                        String::new()
                    } else {
                        format!("{} = ", vars(p, &r.results))
                    };
                    let binds: Vec<String> = r
                        .params
                        .iter()
                        .zip(&r.inputs)
                        .map(|(&q, &i)| format!("{} = {}", var(p, q), var(p, i)))
                        .collect();
                    let _ = writeln!(
                        out,
                        "{pad}  {lhs}role {} x{} warps ({}) {{",
                        r.name,
                        r.warps,
                        binds.join(", ")
                    );
                    block(p, &r.body, depth + 2, out);
                    let _ = writeln!(out, "{pad}  }}");
                }
                let _ = writeln!(out, "{pad}}}");
            }
        }
    }
    if !b.yields.is_empty() {
        let _ = writeln!(out, "{pad}yield {}", vars(p, &b.yields));
    }
}
