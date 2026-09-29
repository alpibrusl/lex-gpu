//! Running an emitted kernel on an NVIDIA GPU.
//!
//! The same shape as [`lex_metal`]: compile the emitted source at load
//! time, allocate buffers, launch. Metal takes MSL text through
//! `newLibraryWithSource:`; the equivalent here is NVRTC, which compiles
//! CUDA C to PTX in-process, and then the driver API loads the PTX and
//! launches it. Neither backend ships a precompiled binary, and neither
//! shells out to a compiler.
//!
//! **The libraries are opened at runtime, not linked.** Linking against
//! `libcuda` makes the *binary* require `libcuda.so.1`, which only a real
//! driver installs — so `cargo test` could not even start the test binary
//! on a machine without a GPU, let alone skip gracefully. The CUDA toolkit
//! ships a stub `libcuda.so` that satisfies the linker and then fails at
//! load time with `cannot open shared object file`, which is a worse
//! failure than the one it looks like it prevents. `dlopen` moves the
//! question from "can this binary run" to "is there a driver here", which
//! is the question actually being asked.
//!
//! So this module builds everywhere and reports "no CUDA device" wherever
//! there is not one. It is still Linux-only, because macOS has no driver
//! to find.
//!
//! Bindings by hand rather than a crate: the surface a kernel launcher
//! needs is about a dozen functions, and both libraries are ABI-stable.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr;
use std::time::Instant;

use lex_ir::{Kernel, Plan, Target};
use lex_msl::program::Lowered;

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type NvrtcResult = c_int;
type NvrtcProgram = *mut c_void;

// `dlopen` has lived in libc since glibc 2.34, so nothing extra is linked.
unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;

unsafe fn open_lib(names: &[&str]) -> Result<*mut c_void, String> {
    for n in names {
        let c = CString::new(*n).expect("no interior nul");
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            return Ok(h);
        }
    }
    let why = unsafe {
        let e = dlerror();
        if e.is_null() {
            String::new()
        } else {
            format!(": {}", CStr::from_ptr(e).to_string_lossy())
        }
    };
    Err(format!("cannot open any of {names:?}{why}"))
}

unsafe fn sym(h: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let c = CString::new(name).expect("no interior nul");
    let p = unsafe { dlsym(h, c.as_ptr()) };
    if p.is_null() {
        return Err(format!("`{name}` is missing from the library"));
    }
    Ok(p)
}

/// Declare a table of function pointers resolved by `dlsym`.
///
/// The transmute is the unavoidable part of any `dlopen` binding: `dlsym`
/// returns an untyped pointer and the signature is asserted here. Each one
/// is checked against the CUDA headers by hand, which is why they are all
/// in one place rather than scattered.
macro_rules! api {
    ($vis:vis struct $name:ident { $( fn $f:ident($($a:ty),* $(,)?) -> $r:ty; )* }) => {
        #[allow(non_snake_case)]
        $vis struct $name { $( $f: unsafe extern "C" fn($($a),*) -> $r, )* }
        impl $name {
            unsafe fn load(h: *mut c_void) -> Result<$name, String> {
                Ok($name {
                    $( $f: unsafe {
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($a),*) -> $r>(
                            sym(h, stringify!($f))?
                        )
                    }, )*
                })
            }
        }
    };
}

api!(struct Driver {
    fn cuInit(c_uint) -> CUresult;
    fn cuDeviceGet(*mut CUdevice, c_int) -> CUresult;
    fn cuDeviceGetName(*mut c_char, c_int, CUdevice) -> CUresult;
    fn cuDeviceGetAttribute(*mut c_int, c_int, CUdevice) -> CUresult;
    fn cuCtxCreate_v2(*mut CUcontext, c_uint, CUdevice) -> CUresult;
    fn cuModuleLoadData(*mut CUmodule, *const c_void) -> CUresult;
    fn cuModuleGetFunction(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
    fn cuMemAlloc_v2(*mut CUdeviceptr, usize) -> CUresult;
    fn cuMemFree_v2(CUdeviceptr) -> CUresult;
    fn cuMemsetD8_v2(CUdeviceptr, u8, usize) -> CUresult;
    fn cuMemcpyHtoD_v2(CUdeviceptr, *const c_void, usize) -> CUresult;
    fn cuMemcpyDtoH_v2(*mut c_void, CUdeviceptr, usize) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuDeviceTotalMem_v2(*mut usize, CUdevice) -> CUresult;
    fn cuGetErrorString(CUresult, *mut *const c_char) -> CUresult;
    fn cuLaunchKernel(
        CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
        c_uint, CUstream, *mut *mut c_void, *mut *mut c_void,
    ) -> CUresult;
    fn cuEventCreate(*mut CUevent, c_uint) -> CUresult;
    fn cuEventRecord(CUevent, CUstream) -> CUresult;
    fn cuEventElapsedTime(*mut f32, CUevent, CUevent) -> CUresult;
    fn cuEventDestroy_v2(CUevent) -> CUresult;
});

api!(pub struct Nvrtc {
    fn nvrtcCreateProgram(
        *mut NvrtcProgram, *const c_char, *const c_char, c_int,
        *const *const c_char, *const *const c_char,
    ) -> NvrtcResult;
    fn nvrtcCompileProgram(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
    fn nvrtcGetPTXSize(NvrtcProgram, *mut usize) -> NvrtcResult;
    fn nvrtcGetPTX(NvrtcProgram, *mut c_char) -> NvrtcResult;
    fn nvrtcGetProgramLogSize(NvrtcProgram, *mut usize) -> NvrtcResult;
    fn nvrtcGetProgramLog(NvrtcProgram, *mut c_char) -> NvrtcResult;
});

/// Compute capability major/minor, as `-arch=compute_XY` wants it.
const ATTR_CC_MAJOR: c_int = 75;
const ATTR_CC_MINOR: c_int = 76;
/// CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK. The static limit, not
/// the opt-in one: the emitter declares `__shared__` arrays, and reaching
/// past 48 KiB needs a dynamic allocation and a host-side opt-in it does
/// not do. `Target::nvidia_ada` reports what the *machine* allows; this is
/// what this backend can currently ask for.
const ATTR_MAX_SHARED_PER_BLOCK: c_int = 8;

fn check(d: &Driver, r: CUresult, what: &str) -> Result<(), String> {
    if r == 0 {
        return Ok(());
    }
    // The driver names its own errors; repeating them here would only go
    // stale.
    let mut s: *const c_char = ptr::null();
    let msg = unsafe {
        if (d.cuGetErrorString)(r, &mut s) == 0 && !s.is_null() {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        } else {
            format!("CUDA error {r}")
        }
    };
    Err(format!("{what}: {msg}"))
}

/// Where `cuda_fp16.h` might be.
///
/// NVRTC compiles from a string in memory and has **no include search path
/// at all** — not even the toolkit's own. `#include <cuda_fp16.h>` fails
/// with "no directories in search list" unless one is supplied, which is
/// not obvious from anything except that message.
fn include_dirs() -> Vec<String> {
    let mut v: Vec<String> = ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|p| format!("{p}/include"))
        .collect();
    v.push("/usr/local/cuda/include".into());
    v.push("/usr/include".into());
    v.retain(|d| std::path::Path::new(d).join("cuda_fp16.h").exists());
    v
}

/// Compile emitted CUDA to PTX.
///
/// Separate from [`Gpu::build_lowered`] because **NVRTC needs no device**:
/// it is a compiler. Splitting it is what lets a machine with the toolkit
/// and no GPU check that the emitted source actually compiles for a real
/// architecture — which is where the missing include path was eventually
/// found, after a rented L4 had to find it instead.
pub fn compile_ptx(rtc: &Nvrtc, source: &str, entry: &str, arch: &str) -> Result<Vec<u8>, String> {
    let src = CString::new(source).map_err(|e| e.to_string())?;
    let unit = CString::new(format!("{entry}.cu")).map_err(|e| e.to_string())?;
    unsafe {
        let mut prog: NvrtcProgram = ptr::null_mut();
        if (rtc.nvrtcCreateProgram)(
            &mut prog,
            src.as_ptr(),
            unit.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        ) != 0
        {
            return Err("nvrtcCreateProgram failed".into());
        }
        let arch = CString::new(format!("--gpu-architecture={arch}")).map_err(|e| e.to_string())?;
        let incs: Vec<CString> = include_dirs()
            .iter()
            .map(|d| CString::new(format!("-I{d}")).expect("no interior nul"))
            .collect();
        let mut opts: Vec<*const c_char> = vec![arch.as_ptr()];
        opts.extend(incs.iter().map(|c| c.as_ptr()));

        if (rtc.nvrtcCompileProgram)(prog, opts.len() as c_int, opts.as_ptr()) != 0 {
            let mut n = 0usize;
            (rtc.nvrtcGetProgramLogSize)(prog, &mut n);
            let mut log = vec![0u8; n.max(1)];
            (rtc.nvrtcGetProgramLog)(prog, log.as_mut_ptr().cast());
            let log = String::from_utf8_lossy(&log)
                .trim_end_matches('\0')
                .to_string();
            let where_ = if incs.is_empty() {
                "\n(no cuda_fp16.h found; set CUDA_HOME)"
            } else {
                ""
            };
            return Err(format!("`{entry}` does not compile:\n{log}{where_}"));
        }
        let mut n = 0usize;
        (rtc.nvrtcGetPTXSize)(prog, &mut n);
        let mut ptx = vec![0u8; n];
        (rtc.nvrtcGetPTX)(prog, ptx.as_mut_ptr().cast());
        Ok(ptx)
    }
}

/// Open NVRTC alone, for compiling without a device.
pub fn nvrtc() -> Result<Nvrtc, String> {
    unsafe {
        Nvrtc::load(open_lib(&[
            "libnvrtc.so",
            "libnvrtc.so.12",
            "/usr/local/cuda/lib64/libnvrtc.so",
        ])?)
    }
}

/// A device buffer. Freed on drop, like Metal's.
pub struct Buffer {
    ptr: CUdeviceptr,
    bytes: usize,
    /// The driver table this was allocated from. A `Buffer` never outlives
    /// its `Gpu`, which owns the loaded library.
    free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
}

impl Buffer {
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.free)(self.ptr);
        }
    }
}

/// A compiled kernel and the launch shape it was lowered for.
pub struct Pipeline {
    f: CUfunction,
    grid: (c_uint, c_uint),
    threads: c_uint,
    shared: c_uint,
    /// Kept alive: the function borrows the module.
    _module: CUmodule,
}

/// What `lex-metal` reports under the same name, so a runtime written
/// against one reads the other without knowing which it has.
pub struct DeviceInfo {
    pub name: String,
    pub unified_memory: bool,
    pub max_threadgroup_bytes: usize,
    pub recommended_working_set_bytes: u64,
}

/// One dispatch of a batch: pipeline, buffers, and optionally fewer blocks
/// (x, y) than the pipeline was lowered for.
pub type Step<'a> = (&'a Pipeline, &'a [&'a Buffer], Option<[usize; 2]>);

pub struct Gpu {
    cu: Driver,
    rtc: Nvrtc,
    dev: CUdevice,
    _ctx: CUcontext,
    name: String,
    arch: String,
    target: Target,
}

impl Gpu {
    pub fn open() -> Result<Gpu, String> {
        unsafe {
            // The driver is `libcuda.so.1`; the toolkit's bare `libcuda.so`
            // is a link-time stub with no implementation behind it.
            let cu = Driver::load(open_lib(&["libcuda.so.1", "libcuda.so"])?)?;
            let rtc = nvrtc()?;

            check(&cu, (cu.cuInit)(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            check(&cu, (cu.cuDeviceGet)(&mut dev, 0), "cuDeviceGet")?;
            let mut ctx: CUcontext = ptr::null_mut();
            check(&cu, (cu.cuCtxCreate_v2)(&mut ctx, 0, dev), "cuCtxCreate")?;

            let mut raw: [c_char; 256] = [0; 256];
            check(
                &cu,
                (cu.cuDeviceGetName)(raw.as_mut_ptr(), raw.len() as c_int, dev),
                "cuDeviceGetName",
            )?;
            let name = CStr::from_ptr(raw.as_ptr()).to_string_lossy().into_owned();

            let (mut major, mut minor) = (0, 0);
            check(
                &cu,
                (cu.cuDeviceGetAttribute)(&mut major, ATTR_CC_MAJOR, dev),
                "compute capability",
            )?;
            check(
                &cu,
                (cu.cuDeviceGetAttribute)(&mut minor, ATTR_CC_MINOR, dev),
                "compute capability",
            )?;
            Ok(Gpu {
                cu,
                rtc,
                dev,
                _ctx: ctx,
                name,
                arch: format!("compute_{major}{minor}"),
                // Hopper's split barriers and TMA are a different lowering
                // from Ada's `cp.async`, so this picks by what the device
                // reports rather than defaulting to the newer one.
                target: if major >= 9 {
                    Target::nvidia_hopper()
                } else {
                    Target::nvidia_ada()
                },
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// What this device compiles for: `compute_89` on an L4.
    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn device(&self) -> CUdevice {
        self.dev
    }

    /// The target the emitter lowers for on this machine.
    pub fn target(&self) -> &Target {
        &self.target
    }

    pub fn info(&self) -> DeviceInfo {
        let attr = |a: c_int| -> usize {
            let mut v: c_int = 0;
            unsafe {
                check(
                    &self.cu,
                    (self.cu.cuDeviceGetAttribute)(&mut v, a, self.dev),
                    "attr",
                )
                .expect("device attribute");
            }
            v as usize
        };
        let mut total: usize = 0;
        unsafe {
            check(
                &self.cu,
                (self.cu.cuDeviceTotalMem_v2)(&mut total, self.dev),
                "cuDeviceTotalMem",
            )
            .expect("total memory");
        }
        DeviceInfo {
            name: self.name.clone(),
            // Discrete: the weights are staged across PCIe, which is the
            // one assumption a Mac-shaped runtime is most likely to have
            // baked in without noticing.
            unified_memory: false,
            max_threadgroup_bytes: attr(ATTR_MAX_SHARED_PER_BLOCK),
            // No "recommended" figure exists here as it does on Metal, so
            // this is the whole device. Anything sizing a cache off it is
            // budgeting against a number with no headroom in it.
            recommended_working_set_bytes: total as u64,
        }
    }

    /// Emit CUDA for `kernel` under `plan`, compile it, and return the
    /// pipeline -- [`lex_metal::Gpu::build`]'s counterpart.
    pub fn build(&self, kernel: &Kernel, plan: &Plan) -> Result<Pipeline, String> {
        let source = crate::emit(kernel, plan, &self.target);
        let ptx = compile_ptx(&self.rtc, &source, &kernel.name, &self.arch)?;
        let [gx, gy, _] = plan.launch.threadgroups;
        let [tx, _, _] = plan.launch.threads_per_threadgroup;
        self.module(&ptx, &kernel.name, (gx, gy), tx)
    }

    /// Compile emitted CUDA for *this* device and take its entry point.
    ///
    /// NVRTC targets the device actually present rather than a fixed
    /// architecture, which is the point of compiling at load time: the same
    /// emitted source runs on whatever the machine has.
    pub fn build_lowered(&self, lowered: &Lowered) -> Result<Pipeline, String> {
        let ptx = compile_ptx(&self.rtc, &lowered.source, &lowered.entry, &self.arch)?;
        self.module(
            &ptx,
            &lowered.entry,
            (lowered.grid, lowered.grid2.max(1)),
            lowered.threads,
        )
    }

    /// Load PTX and take its entry point.
    ///
    /// No dynamic shared memory. Both emitters declare every threadgroup
    /// array statically (`__shared__ float scratch[..]`), so a byte count
    /// passed at launch as well was reserved *on top* of those arrays and
    /// never used: each kernel held its shared memory twice, which for the
    /// ones with real tiles -- attention, the GEMM -- halves how many of
    /// them fit on an SM.
    fn module(
        &self,
        ptx: &[u8],
        entry: &str,
        grid: (usize, usize),
        threads: usize,
    ) -> Result<Pipeline, String> {
        let shared = 0usize;
        unsafe {
            let mut module: CUmodule = ptr::null_mut();
            check(
                &self.cu,
                (self.cu.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
            let name = CString::new(entry).map_err(|e| e.to_string())?;
            let mut f: CUfunction = ptr::null_mut();
            check(
                &self.cu,
                (self.cu.cuModuleGetFunction)(&mut f, module, name.as_ptr()),
                "cuModuleGetFunction",
            )?;
            Ok(Pipeline {
                f,
                grid: (grid.0 as c_uint, grid.1.max(1) as c_uint),
                threads: threads as c_uint,
                shared: shared as c_uint,
                _module: module,
            })
        }
    }

    pub fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        let bytes = std::mem::size_of_val(data);
        let b = self.alloc(bytes);
        unsafe {
            check(
                &self.cu,
                (self.cu.cuMemcpyHtoD_v2)(b.ptr, data.as_ptr().cast(), bytes),
                "cuMemcpyHtoD",
            )
            .expect("upload");
        }
        b
    }

    pub fn zeroed<T: Copy>(&self, len: usize) -> Buffer {
        let bytes = len * std::mem::size_of::<T>();
        let b = self.alloc(bytes);
        unsafe {
            check(
                &self.cu,
                (self.cu.cuMemsetD8_v2)(b.ptr, 0, bytes),
                "cuMemsetD8",
            )
            .expect("zero");
        }
        b
    }

    fn alloc(&self, bytes: usize) -> Buffer {
        let mut ptr_: CUdeviceptr = 0;
        unsafe {
            check(
                &self.cu,
                (self.cu.cuMemAlloc_v2)(&mut ptr_, bytes.max(1)),
                "cuMemAlloc",
            )
            .expect("allocate");
        }
        Buffer {
            ptr: ptr_,
            bytes: bytes.max(1),
            free: self.cu.cuMemFree_v2,
        }
    }

    pub fn write<T: Copy>(&self, buf: &Buffer, offset: usize, data: &[T]) {
        let sz = std::mem::size_of::<T>();
        let end = (offset + data.len()) * sz;
        assert!(end <= buf.bytes, "write of {end} B into {} B", buf.bytes);
        unsafe {
            check(
                &self.cu,
                (self.cu.cuMemcpyHtoD_v2)(
                    buf.ptr + (offset * sz) as u64,
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data),
                ),
                "cuMemcpyHtoD",
            )
            .expect("write");
        }
    }

    pub fn download<T: Copy>(&self, buf: &Buffer, out: &mut [T]) {
        let bytes = std::mem::size_of_val(out);
        assert!(bytes <= buf.bytes, "read of {bytes} B from {} B", buf.bytes);
        unsafe {
            check(&self.cu, (self.cu.cuCtxSynchronize)(), "cuCtxSynchronize").expect("sync");
            check(
                &self.cu,
                (self.cu.cuMemcpyDtoH_v2)(out.as_mut_ptr().cast(), buf.ptr, bytes),
                "cuMemcpyDtoH",
            )
            .expect("download");
        }
    }

    pub fn download_at<T: Copy>(&self, buf: &Buffer, offset: usize, out: &mut [T]) {
        let sz = std::mem::size_of::<T>();
        let bytes = std::mem::size_of_val(out);
        assert!(
            offset * sz + bytes <= buf.bytes,
            "read of {bytes} B at {} B from {} B",
            offset * sz,
            buf.bytes
        );
        unsafe {
            check(&self.cu, (self.cu.cuCtxSynchronize)(), "cuCtxSynchronize").expect("sync");
            check(
                &self.cu,
                (self.cu.cuMemcpyDtoH_v2)(
                    out.as_mut_ptr().cast(),
                    buf.ptr + (offset * sz) as u64,
                    bytes,
                ),
                "cuMemcpyDtoH",
            )
            .expect("download");
        }
    }

    /// Run a sequence of dispatches and wait once.
    ///
    /// Metal needs an explicit barrier between a write and a dependent
    /// read because its encoder is concurrent. The default CUDA stream
    /// orders its launches, so correctness here is free and the
    /// *overlap* is what is missing: independent kernels that share the
    /// GPU on Metal run one after another here. Fixing that means several
    /// streams and events, and it should be done against a measurement
    /// rather than on principle.
    pub fn run_all(&self, steps: &[(&Pipeline, &[&Buffer])]) {
        let with: Vec<Step<'_>> = steps.iter().map(|&(p, b)| (p, b, None)).collect();
        self.run_launches(&with);
    }

    /// [`Gpu::run_all`] where a step may launch fewer blocks than it was
    /// lowered for. Launching more than planned is refused, as on Metal.
    ///
    /// Returns (CPU seconds spent launching, wall seconds to the sync).
    /// The second is *not* Metal's device-timed figure: it is the wall
    /// clock around one `cuCtxSynchronize`, so it includes the launch
    /// overhead and is only comparable to itself.
    pub fn run_launches(&self, steps: &[Step<'_>]) -> (f64, f64) {
        let t0 = Instant::now();
        for &(p, buffers, groups) in steps {
            let (gx, gy) = match groups {
                None => (p.grid.0, p.grid.1),
                Some([x, y]) => {
                    assert!(
                        x as c_uint <= p.grid.0 && y as c_uint <= p.grid.1,
                        "launch of {x}x{y} over a plan of {}x{}",
                        p.grid.0,
                        p.grid.1
                    );
                    (x as c_uint, y as c_uint)
                }
            };
            self.launch(p, buffers, gx, gy).expect("cuLaunchKernel");
        }
        let cpu = t0.elapsed().as_secs_f64();
        unsafe {
            check(&self.cu, (self.cu.cuCtxSynchronize)(), "cuCtxSynchronize").expect("sync");
        }
        (cpu, t0.elapsed().as_secs_f64())
    }

    /// Run `steps` in order and return each one's GPU time, in seconds.
    ///
    /// An event goes on the stream before the first launch and after each
    /// one, and nothing waits until the end, so the launches queue exactly
    /// as in [`Gpu::run_launches`] -- unlike synchronising after each,
    /// which idles the GPU for a host round trip per kernel and was
    /// measured to make per-kernel times sum to twice the real step. On one
    /// in-order stream the interval between two events is the kernel
    /// between them plus any time the GPU waited for the host to launch it,
    /// and that wait is a real cost of the normal path too.
    pub fn run_each_timed(&self, steps: &[Step<'_>]) -> Vec<f64> {
        let d = &self.cu;
        let mut ev: Vec<CUevent> = vec![ptr::null_mut(); steps.len() + 1];
        unsafe {
            for e in &mut ev {
                check(d, (d.cuEventCreate)(e, 0), "cuEventCreate").expect("event");
            }
            check(
                d,
                (d.cuEventRecord)(ev[0], ptr::null_mut()),
                "cuEventRecord",
            )
            .expect("record");
        }
        for (i, &(p, buffers, groups)) in steps.iter().enumerate() {
            let (gx, gy) = match groups {
                None => (p.grid.0, p.grid.1),
                Some([x, y]) => {
                    assert!(
                        x as c_uint <= p.grid.0 && y as c_uint <= p.grid.1,
                        "launch of {x}x{y} over a plan of {}x{}",
                        p.grid.0,
                        p.grid.1
                    );
                    (x as c_uint, y as c_uint)
                }
            };
            self.launch(p, buffers, gx, gy).expect("cuLaunchKernel");
            unsafe {
                check(
                    d,
                    (d.cuEventRecord)(ev[i + 1], ptr::null_mut()),
                    "cuEventRecord",
                )
                .expect("record");
            }
        }
        let mut out = Vec::with_capacity(steps.len());
        unsafe {
            check(d, (d.cuCtxSynchronize)(), "cuCtxSynchronize").expect("sync");
            for w in ev.windows(2) {
                let mut ms = 0f32;
                check(
                    d,
                    (d.cuEventElapsedTime)(&mut ms, w[0], w[1]),
                    "cuEventElapsedTime",
                )
                .expect("elapsed");
                out.push(ms as f64 / 1e3);
            }
            for e in ev {
                (d.cuEventDestroy_v2)(e);
            }
        }
        out
    }

    /// One launch, no synchronise.
    fn launch(
        &self,
        p: &Pipeline,
        buffers: &[&Buffer],
        gx: c_uint,
        gy: c_uint,
    ) -> Result<(), String> {
        let mut ptrs: Vec<CUdeviceptr> = buffers.iter().map(|b| b.ptr).collect();
        let mut params: Vec<*mut c_void> = ptrs
            .iter_mut()
            .map(|p| (p as *mut CUdeviceptr).cast())
            .collect();
        unsafe {
            check(
                &self.cu,
                (self.cu.cuLaunchKernel)(
                    p.f,
                    gx,
                    gy,
                    1,
                    p.threads,
                    1,
                    1,
                    p.shared,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    /// Launch, binding `buffers` in order.
    ///
    /// CUDA takes parameters positionally, so the order here *is* the
    /// binding — there is no index to get wrong, and equally none to check.
    /// Metal's `[[buffer(n)]]` would have caught a mismatch at pipeline
    /// creation; here a mis-ordered call reads the wrong memory and returns
    /// plausible numbers, which is why the tests compare against the
    /// interpreter rather than eyeballing output.
    pub fn run(&self, p: &Pipeline, buffers: &[&Buffer]) -> Result<(), String> {
        let mut ptrs: Vec<CUdeviceptr> = buffers.iter().map(|b| b.ptr).collect();
        let mut params: Vec<*mut c_void> = ptrs
            .iter_mut()
            .map(|p| (p as *mut CUdeviceptr).cast())
            .collect();
        unsafe {
            check(
                &self.cu,
                (self.cu.cuLaunchKernel)(
                    p.f,
                    p.grid.0,
                    p.grid.1,
                    1,
                    p.threads,
                    1,
                    1,
                    p.shared,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            check(&self.cu, (self.cu.cuCtxSynchronize)(), "cuCtxSynchronize")
        }
    }
}
