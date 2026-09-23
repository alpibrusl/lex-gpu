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

use lex_msl::program::Lowered;

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;
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
    fn cuGetErrorString(CUresult, *mut *const c_char) -> CUresult;
    fn cuLaunchKernel(
        CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
        c_uint, CUstream, *mut *mut c_void, *mut *mut c_void,
    ) -> CUresult;
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

pub struct Gpu {
    cu: Driver,
    rtc: Nvrtc,
    dev: CUdevice,
    _ctx: CUcontext,
    name: String,
    arch: String,
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

    /// Compile emitted CUDA for *this* device and take its entry point.
    ///
    /// NVRTC targets the device actually present rather than a fixed
    /// architecture, which is the point of compiling at load time: the same
    /// emitted source runs on whatever the machine has.
    pub fn build_lowered(&self, lowered: &Lowered) -> Result<Pipeline, String> {
        let ptx = compile_ptx(&self.rtc, &lowered.source, &lowered.entry, &self.arch)?;
        unsafe {
            let mut module: CUmodule = ptr::null_mut();
            check(
                &self.cu,
                (self.cu.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
            let entry = CString::new(lowered.entry.as_str()).map_err(|e| e.to_string())?;
            let mut f: CUfunction = ptr::null_mut();
            check(
                &self.cu,
                (self.cu.cuModuleGetFunction)(&mut f, module, entry.as_ptr()),
                "cuModuleGetFunction",
            )?;
            Ok(Pipeline {
                f,
                grid: (lowered.grid as c_uint, lowered.grid2.max(1) as c_uint),
                threads: lowered.threads as c_uint,
                shared: lowered.threadgroup_bytes as c_uint,
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
                    data.len() * sz,
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
