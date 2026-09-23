use std::time::Instant;

use lex_ir::{Kernel, Launch, Plan, Target};
use lex_msl::program::Lowered;
use metal::objc::rc::autoreleasepool;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLDispatchType,
    MTLResourceOptions, MTLSize,
};

pub struct DeviceInfo {
    pub name: String,
    pub unified_memory: bool,
    pub max_threadgroup_bytes: usize,
    pub recommended_working_set_bytes: u64,
}

/// One dispatch of a batch: pipeline, buffers, and optionally fewer
/// threadgroups (x, y) than the pipeline's plan.
pub type Step<'a> = (&'a Pipeline, &'a [&'a Buffer], Option<[usize; 2]>);

/// An open Metal device plus its command queue.
pub struct Gpu {
    device: Device,
    queue: CommandQueue,
    target: Target,
}

/// A compiled kernel, bound to the plan it was emitted for.
///
/// The plan travels with the pipeline so that dispatch geometry can never drift
/// from the geometry the source was generated against -- the class of bug that
/// shows up as a wrong answer in one corner of a tensor.
pub struct Pipeline {
    pso: ComputePipelineState,
    plan: Plan,
    /// Which bindings the kernel writes, when known (lowered programs).
    /// `None` is treated as writing every buffer it is given.
    writes: Option<Vec<bool>>,
    pub name: String,
    pub source: String,
}

impl Pipeline {
    /// Threads the hardware runs in lockstep for this pipeline. Worth checking
    /// against the target table: if Apple ever ships a non-32 simd width, the
    /// reduction in the emitted RMSNorm is wrong and this is how we find out.
    pub fn thread_execution_width(&self) -> usize {
        self.pso.thread_execution_width() as usize
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.pso.max_total_threads_per_threadgroup() as usize
    }
}

impl Gpu {
    pub fn open() -> Result<Gpu, String> {
        let device = Device::system_default().ok_or("no Metal device found")?;
        let queue = device.new_command_queue();
        Ok(Gpu {
            device,
            queue,
            target: Target::apple_m_series(),
        })
    }

    pub fn target(&self) -> &Target {
        &self.target
    }

    pub fn info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name().to_string(),
            unified_memory: self.device.has_unified_memory(),
            max_threadgroup_bytes: self.device.max_threadgroup_memory_length() as usize,
            recommended_working_set_bytes: self.device.recommended_max_working_set_size(),
        }
    }

    /// Emit MSL for `kernel` under `plan`, compile it, and return the pipeline.
    ///
    /// Compilation happens at run time from source. That is the right tradeoff
    /// while iterating -- Metal's compiler is fast and the error messages come
    /// back with line numbers into text we generated -- and it is not the
    /// shipping story, which is a precompiled metallib.
    pub fn build(&self, kernel: &Kernel, plan: &Plan) -> Result<Pipeline, String> {
        let source = lex_msl::emit(kernel, plan, &self.target);
        self.pipeline(&kernel.name, source, plan, true)
    }

    /// Compile a lowered `lex-front` program.
    ///
    /// Fast math is off: typed kernels rely on IEEE infinities (an online
    /// softmax starts its running max at -inf), which fast math is allowed to
    /// assume away.
    pub fn build_lowered(&self, lowered: &Lowered) -> Result<Pipeline, String> {
        let plan = Plan {
            launch: Launch {
                threadgroups: [lowered.grid, lowered.grid2, 1],
                threads_per_threadgroup: [lowered.threads, 1, 1],
            },
            threadgroup_bytes: lowered.threadgroup_bytes,
            vec_width: 1,
            threads_per_tg: lowered.threads,
            simdgroups_per_tg: lowered.threads.div_ceil(self.target.simd_width),
            vec_lanes: 0,
        };
        let mut p = self.pipeline(&lowered.entry, lowered.source.clone(), &plan, false)?;
        p.writes = Some(lowered.writes.clone());
        Ok(p)
    }

    fn pipeline(
        &self,
        name: &str,
        source: String,
        plan: &Plan,
        fast_math: bool,
    ) -> Result<Pipeline, String> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(fast_math);
        let library = self
            .device
            .new_library_with_source(&source, &options)
            .map_err(|e| {
                format!("MSL compile failed for `{name}`:\n{e}\n--- source ---\n{source}")
            })?;
        let function = library
            .get_function(name, None)
            .map_err(|e| format!("no function `{name}` in compiled library: {e}"))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("pipeline creation failed for `{name}`: {e}"))?;

        let want = plan.threads_per_tg;
        let allowed = pso.max_total_threads_per_threadgroup() as usize;
        if want > allowed {
            return Err(format!(
                "plan asks for {want} threads per threadgroup, `{name}` allows {allowed}"
            ));
        }

        Ok(Pipeline {
            pso,
            plan: *plan,
            writes: None,
            name: name.to_string(),
            source,
        })
    }

    /// Upload a slice into a shared-storage buffer.
    ///
    /// `StorageModeShared` on unified memory means this is a plain allocation
    /// the CPU and GPU both address -- no staging copy, which is the one
    /// user-visible simplification unified memory buys.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn zeroed<T: Copy>(&self, len: usize) -> Buffer {
        let bytes = (len * std::mem::size_of::<T>()) as u64;
        let buf = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(buf.contents().cast::<u8>(), 0, bytes as usize) };
        buf
    }

    /// Overwrite elements `offset..offset + data.len()` of a buffer.
    ///
    /// Same contract as [`Gpu::download`]: nothing may be running that reads
    /// or writes the buffer, and `T` must be the element type the kernels use.
    pub fn write<T: Copy>(&self, buf: &Buffer, offset: usize, data: &[T]) {
        let end = (offset + data.len()) * std::mem::size_of::<T>();
        assert!(
            end as u64 <= buf.length(),
            "write past the end of a {} B buffer",
            buf.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents().cast::<T>().add(offset),
                data.len(),
            )
        };
    }

    /// Copy a buffer's contents back into host memory.
    ///
    /// # Safety contract
    /// The caller must have synchronised (i.e. the command buffer that wrote
    /// this buffer completed) and `out` must match the element type the kernel
    /// wrote. Both hold for the bench; a real runtime gets a typed handle.
    pub fn download<T: Copy>(&self, buf: &Buffer, out: &mut [T]) {
        let bytes = std::mem::size_of_val(out);
        assert!(
            bytes as u64 <= buf.length(),
            "download of {bytes} B from a {} B buffer",
            buf.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(buf.contents().cast::<T>(), out.as_mut_ptr(), out.len())
        };
    }

    /// Read `out.len()` elements starting `offset` elements in.
    ///
    /// Prefill wants the last row of a `[tokens, vocab]` logit buffer and
    /// nothing else. Without an offset the caller copies the whole thing
    /// and discards all but the tail -- 7.9 MB a chunk on this model, most
    /// of a gigabyte over a long prompt.
    pub fn download_at<T: Copy>(&self, buf: &Buffer, offset: usize, out: &mut [T]) {
        let sz = std::mem::size_of::<T>();
        let end = (offset + out.len()) * sz;
        assert!(
            end as u64 <= buf.length(),
            "read of {end} B from a {} B buffer",
            buf.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents().cast::<T>().add(offset),
                out.as_mut_ptr(),
                out.len(),
            )
        };
    }

    /// Run the pipeline once and wait.
    pub fn run(&self, pipeline: &Pipeline, buffers: &[&Buffer]) {
        self.dispatch(pipeline, buffers, 1);
    }

    /// Fastest observed time for a single dispatch, in seconds.
    ///
    /// `iters` dispatches are encoded into one command buffer so that encode
    /// and submit cost is amortised rather than measured, and the whole thing
    /// is repeated `repeats` times with the minimum taken -- the minimum is the
    /// right statistic here because every source of noise (thermal, contention,
    /// other processes) can only make a run slower.
    pub fn time(
        &self,
        pipeline: &Pipeline,
        buffers: &[&Buffer],
        iters: usize,
        repeats: usize,
    ) -> f64 {
        assert!(iters > 0 && repeats > 0);
        // Warm up: first dispatch pays for residency and any lazy compilation.
        self.dispatch(pipeline, buffers, 2);

        let mut best = f64::INFINITY;
        for _ in 0..repeats {
            let start = Instant::now();
            self.dispatch(pipeline, buffers, iters);
            let elapsed = start.elapsed().as_secs_f64();
            best = best.min(elapsed / iters as f64);
        }
        best
    }

    /// Run one dispatch and return how long the GPU spent on it, in seconds,
    /// from the command buffer's own timestamps — without the CPU's submit
    /// and wait, which dominate wall time for small kernels.
    pub fn run_gpu_timed(&self, pipeline: &Pipeline, buffers: &[&Buffer]) -> f64 {
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let l = pipeline.plan.launch;
            enc.set_compute_pipeline_state(&pipeline.pso);
            for (i, &b) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(
                    l.threadgroups[0] as u64,
                    l.threadgroups[1] as u64,
                    l.threadgroups[2] as u64,
                ),
                MTLSize::new(
                    l.threads_per_threadgroup[0] as u64,
                    l.threads_per_threadgroup[1] as u64,
                    l.threads_per_threadgroup[2] as u64,
                ),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let (start, end) = gpu_times(cb);
            end - start
        })
    }

    /// Run a sequence of dispatches in one command buffer and wait once.
    ///
    /// Same semantics as calling [`Gpu::run`] for each in turn, without a
    /// CPU round trip between them. The encoder is concurrent: dispatches
    /// may overlap, and a buffer-scope memory barrier is placed only before
    /// a dispatch that reads or writes a buffer an unbarriered earlier one
    /// wrote, or writes one it read (what llama.cpp's Metal backend does).
    /// Independent kernels (the q, k and v matvecs, gate and up, the small
    /// ones) then share the GPU instead of each paying a drain.
    /// `LEX_SERIAL=1` falls back to a serial encoder.
    pub fn run_all(&self, steps: &[(&Pipeline, &[&Buffer])]) {
        self.run_all_timed(steps);
    }

    /// [`Gpu::run_all`], returning (CPU seconds spent encoding, GPU seconds
    /// executing).
    pub fn run_all_timed(&self, steps: &[(&Pipeline, &[&Buffer])]) -> (f64, f64) {
        let with: Vec<Step<'_>> = steps.iter().map(|&(p, b)| (p, b, None)).collect();
        self.run_launches(&with)
    }

    /// [`Gpu::run_all_timed`] where a step may launch fewer threadgroups
    /// than its plan along x and y: a kernel whose instances are
    /// independent (splits of a KV cache, say) run only as many as the
    /// sequence needs. Launching more than planned is refused.
    pub fn run_launches(&self, steps: &[Step<'_>]) -> (f64, f64) {
        autoreleasepool(|| {
            let t0 = Instant::now();
            let cb = self.queue.new_command_buffer();
            let serial = std::env::var_os("LEX_SERIAL").is_some();
            let enc = if serial {
                cb.new_compute_command_encoder()
            } else {
                cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)
            };
            // Buffers written / read since the last barrier, by GPU address.
            let (mut written, mut read) = (Vec::<u64>::new(), Vec::<u64>::new());
            for &(pipeline, buffers, groups) in steps {
                if !serial {
                    let mut w = vec![];
                    let mut r = vec![];
                    for (i, b) in buffers.iter().enumerate() {
                        let writes = pipeline
                            .writes
                            .as_ref()
                            .is_none_or(|v| v.get(i) != Some(&false));
                        if writes {
                            w.push(b.gpu_address())
                        } else {
                            r.push(b.gpu_address())
                        }
                    }
                    let hazard = r.iter().chain(&w).any(|a| written.contains(a))
                        || w.iter().any(|a| read.contains(a));
                    if hazard {
                        barrier(enc);
                        written.clear();
                        read.clear();
                    }
                    written.extend(w);
                    read.extend(r);
                }
                let l = pipeline.plan.launch;
                let [gx, gy] = groups.unwrap_or([l.threadgroups[0], l.threadgroups[1]]);
                assert!(
                    gx <= l.threadgroups[0] && gy <= l.threadgroups[1],
                    "launch of {gx}x{gy} threadgroups exceeds `{}`'s planned {:?}",
                    pipeline.name,
                    l.threadgroups
                );
                enc.set_compute_pipeline_state(&pipeline.pso);
                for (i, &b) in buffers.iter().enumerate() {
                    enc.set_buffer(i as u64, Some(b), 0);
                }
                enc.dispatch_thread_groups(
                    MTLSize::new(gx as u64, gy as u64, l.threadgroups[2] as u64),
                    MTLSize::new(
                        l.threads_per_threadgroup[0] as u64,
                        l.threads_per_threadgroup[1] as u64,
                        l.threads_per_threadgroup[2] as u64,
                    ),
                );
            }
            enc.end_encoding();
            let encode = t0.elapsed().as_secs_f64();
            cb.commit();
            cb.wait_until_completed();
            let (start, end) = gpu_times(cb);
            (encode, end - start)
        })
    }

    fn dispatch(&self, pipeline: &Pipeline, buffers: &[&Buffer], iters: usize) {
        let l = pipeline.plan.launch;
        let groups = MTLSize::new(
            l.threadgroups[0] as u64,
            l.threadgroups[1] as u64,
            l.threadgroups[2] as u64,
        );
        let threads = MTLSize::new(
            l.threads_per_threadgroup[0] as u64,
            l.threads_per_threadgroup[1] as u64,
            l.threads_per_threadgroup[2] as u64,
        );

        // Without a pool, every command buffer and encoder in the loop is
        // leaked until the process exits -- which on a 1 GiB working set is
        // noticeable within seconds.
        autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline.pso);
            for (i, &b) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            // A compute encoder dispatches serially by default, so N encoded
            // dispatches take N kernel times rather than overlapping.
            for _ in 0..iters {
                enc.dispatch_thread_groups(groups, threads);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });
    }
}

/// A completed command buffer's GPU start and end times, in seconds.
/// metal-rs does not wrap these properties.
/// `memoryBarrierWithScope:MTLBarrierScopeBuffers` on a concurrent encoder.
fn barrier(enc: &metal::ComputeCommandEncoderRef) {
    use metal::objc::{msg_send, sel, sel_impl};
    const MTL_BARRIER_SCOPE_BUFFERS: u64 = 1;
    unsafe {
        let () = msg_send![enc, memoryBarrierWithScope: MTL_BARRIER_SCOPE_BUFFERS];
    }
}

fn gpu_times(cb: &metal::CommandBufferRef) -> (f64, f64) {
    use metal::objc::{msg_send, sel, sel_impl};
    unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) }
}
