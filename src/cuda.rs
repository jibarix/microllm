//! The NVIDIA GPU through the CUDA driver API, loaded at run time from the driver's own
//! library (nvcuda.dll / libcuda.so.1), so the binary builds and runs without CUDA and
//! needs no crates. The kernels are src/kernels.cu, compiled to PTX by build.rs and
//! embedded here; the driver compiles the PTX for whatever GPU it finds.
//!
//! One device, one context, the default stream: every kernel runs in launch order and a
//! copy back to the host waits for all of them.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

type CuResult = i32;
type Ptr = u64;

#[cfg(windows)]
mod dl {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }
    pub const LIB: &str = "nvcuda.dll\0";
    pub unsafe fn open(name: &str) -> *mut c_void {
        unsafe { LoadLibraryA(name.as_ptr()) }
    }
    pub unsafe fn sym(lib: *mut c_void, name: &str) -> *mut c_void {
        unsafe { GetProcAddress(lib, name.as_ptr()) }
    }
}
#[cfg(unix)]
mod dl {
    use std::ffi::c_void;
    unsafe extern "C" {
        fn dlopen(name: *const u8, flags: i32) -> *mut c_void;
        fn dlsym(lib: *mut c_void, name: *const u8) -> *mut c_void;
    }
    pub const LIB: &str = "libcuda.so.1\0";
    pub unsafe fn open(name: &str) -> *mut c_void {
        unsafe { dlopen(name.as_ptr(), 2) }
    }
    pub unsafe fn sym(lib: *mut c_void, name: &str) -> *mut c_void {
        unsafe { dlsym(lib, name.as_ptr()) }
    }
}

/// The driver functions used, by their versioned export names.
struct Api {
    init: unsafe extern "system" fn(u32) -> CuResult,
    device_get: unsafe extern "system" fn(*mut i32, i32) -> CuResult,
    device_get_name: unsafe extern "system" fn(*mut c_char, i32, i32) -> CuResult,
    primary_ctx_retain: unsafe extern "system" fn(*mut *mut c_void, i32) -> CuResult,
    ctx_set_current: unsafe extern "system" fn(*mut c_void) -> CuResult,
    ctx_synchronize: unsafe extern "system" fn() -> CuResult,
    module_load_data_ex: unsafe extern "system" fn(*mut *mut c_void, *const c_void, u32, *mut i32, *mut *mut c_void) -> CuResult,
    module_get_function: unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CuResult,
    mem_alloc: unsafe extern "system" fn(*mut Ptr, usize) -> CuResult,
    memcpy_htod: unsafe extern "system" fn(Ptr, *const c_void, usize) -> CuResult,
    memcpy_dtoh: unsafe extern "system" fn(*mut c_void, Ptr, usize) -> CuResult,
    memset_d32: unsafe extern "system" fn(Ptr, u32, usize) -> CuResult,
    launch_kernel: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CuResult,
    get_error_name: unsafe extern "system" fn(CuResult, *mut *const c_char) -> CuResult,
}

pub struct Device {
    api: Api,
    ctx: *mut c_void,
    funcs: HashMap<String, *mut c_void>,
    pub name: String,
    /// Freed buffers by size in bytes, reused by the next allocation of that size: every
    /// training step builds the same shapes, so after the first step nothing is allocated.
    pool: Mutex<HashMap<usize, Vec<Ptr>>>,
}
// The context and function handles are process-wide; `dev()` makes the context current on
// the calling thread before handing the device out.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

static DEVICE: OnceLock<Result<Device, String>> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Open the GPU (once) and route forward passes to it (see `model::forward_eval`).
pub fn enable() -> Result<&'static str, String> {
    let d = DEVICE.get_or_init(open).as_ref().map_err(|e| e.clone())?;
    ENABLED.store(true, Ordering::SeqCst);
    Ok(&d.name)
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

/// How the embedded kernels were built (the nvcc release, or why they were not).
pub fn ptx_build() -> &'static str {
    env!("MICROLLM_PTX_BUILD")
}

pub fn dev() -> &'static Device {
    let d = DEVICE.get().and_then(|d| d.as_ref().ok()).expect("GPU used before cuda::enable()");
    d.check(unsafe { (d.api.ctx_set_current)(d.ctx) }, "cuCtxSetCurrent");
    d
}

fn open() -> Result<Device, String> {
    if PTX.is_empty() {
        return Err(format!("this binary was built without GPU kernels ({})", ptx_build()));
    }
    unsafe {
        let lib = dl::open(dl::LIB);
        if lib.is_null() {
            return Err(format!("could not load the NVIDIA driver library {}", dl::LIB.trim_end_matches('\0')));
        }
        macro_rules! f {
            ($name:literal) => {{
                let p = dl::sym(lib, concat!($name, "\0"));
                if p.is_null() {
                    return Err(format!("the NVIDIA driver has no {}", $name));
                }
                std::mem::transmute(p)
            }};
        }
        let api = Api {
            init: f!("cuInit"),
            device_get: f!("cuDeviceGet"),
            device_get_name: f!("cuDeviceGetName"),
            primary_ctx_retain: f!("cuDevicePrimaryCtxRetain"),
            ctx_set_current: f!("cuCtxSetCurrent"),
            ctx_synchronize: f!("cuCtxSynchronize"),
            module_load_data_ex: f!("cuModuleLoadDataEx"),
            module_get_function: f!("cuModuleGetFunction"),
            mem_alloc: f!("cuMemAlloc_v2"),
            memcpy_htod: f!("cuMemcpyHtoD_v2"),
            memcpy_dtoh: f!("cuMemcpyDtoH_v2"),
            memset_d32: f!("cuMemsetD32_v2"),
            launch_kernel: f!("cuLaunchKernel"),
            get_error_name: f!("cuGetErrorName"),
        };
        let err = |what: &str, r: CuResult| format!("{what} failed with CUDA error {r}");
        let r = (api.init)(0);
        if r != 0 {
            return Err(err("cuInit (no usable NVIDIA GPU?)", r));
        }
        let mut device = 0;
        let r = (api.device_get)(&mut device, 0);
        if r != 0 {
            return Err(err("cuDeviceGet", r));
        }
        let mut name = [0 as c_char; 256];
        (api.device_get_name)(name.as_mut_ptr(), 256, device);
        let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
        let mut ctx = std::ptr::null_mut();
        let r = (api.primary_ctx_retain)(&mut ctx, device);
        if r != 0 {
            return Err(err("cuDevicePrimaryCtxRetain", r));
        }
        (api.ctx_set_current)(ctx);
        let ptx = CString::new(PTX).unwrap();
        let mut module = std::ptr::null_mut();
        let r = (api.module_load_data_ex)(&mut module, ptx.as_ptr() as *const c_void, 0, std::ptr::null_mut(), std::ptr::null_mut());
        if r != 0 {
            return Err(err("loading the kernels (cuModuleLoadDataEx)", r));
        }
        let mut funcs = HashMap::new();
        for line in PTX.lines() {
            if let Some(n) = line.trim().strip_prefix(".visible .entry ") {
                let n = n.trim_end_matches('(').trim();
                let mut f = std::ptr::null_mut();
                let c = CString::new(n).unwrap();
                let r = (api.module_get_function)(&mut f, module, c.as_ptr());
                if r != 0 {
                    return Err(err(&format!("cuModuleGetFunction({n})"), r));
                }
                funcs.insert(n.to_string(), f);
            }
        }
        Ok(Device { api, ctx, funcs, name, pool: Mutex::new(HashMap::new()) })
    }
}

/// A kernel argument: a device pointer, an int or a float (passed by address, as CUDA wants).
#[derive(Clone, Copy)]
pub enum Arg {
    P(Ptr),
    I(i32),
    L(i64),
    F(f32),
    D(f64),
}

impl Device {
    fn check(&self, r: CuResult, what: &str) {
        if r != 0 {
            let mut s: *const c_char = std::ptr::null();
            unsafe { (self.api.get_error_name)(r, &mut s) };
            let name = if s.is_null() { "unknown".into() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() };
            panic!("CUDA {what} failed: {name} ({r})");
        }
    }

    pub fn launch(&self, kernel: &str, grid: (u32, u32, u32), block: (u32, u32, u32), args: &[Arg]) {
        let f = *self.funcs.get(kernel).unwrap_or_else(|| panic!("no GPU kernel {kernel}"));
        // Each argument as 8 bytes; the kernel reads 4 for ints and floats (little-endian).
        let mut store: Vec<u64> = args
            .iter()
            .map(|a| match *a {
                Arg::P(p) => p,
                Arg::I(i) => i as u32 as u64,
                Arg::L(l) => l as u64,
                Arg::F(x) => x.to_bits() as u64,
                Arg::D(x) => x.to_bits(),
            })
            .collect();
        let mut ptrs: Vec<*mut c_void> = store.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
        let r = unsafe { (self.api.launch_kernel)(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, 0, std::ptr::null_mut(), ptrs.as_mut_ptr(), std::ptr::null_mut()) };
        self.check(r, kernel);
    }

    /// One thread per element, 256 per block.
    pub fn launch1(&self, kernel: &str, n: usize, args: &[Arg]) {
        if n > 0 {
            self.launch(kernel, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), args);
        }
    }

    /// One warp per row, 8 rows per block.
    pub fn launch_rows(&self, kernel: &str, rows: usize, args: &[Arg]) {
        if rows > 0 {
            self.launch(kernel, (rows.div_ceil(8) as u32, 1, 1), (256, 1, 1), args);
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn sync(&self) {
        self.check(unsafe { (self.api.ctx_synchronize)() }, "cuCtxSynchronize (an earlier kernel failed?)");
    }

    fn alloc(&self, bytes: usize) -> Ptr {
        if let Some(p) = self.pool.lock().unwrap().get_mut(&bytes).and_then(|v| v.pop()) {
            return p;
        }
        let mut p = 0;
        self.check(unsafe { (self.api.mem_alloc)(&mut p, bytes.max(4)) }, "cuMemAlloc");
        p
    }
}

/// `len` 32-bit numbers (f32 or i32) on the GPU. Dropping it returns the memory to the pool.
pub struct Buf {
    pub ptr: Ptr,
    pub len: usize,
    /// False for a view (`Buf::view`), which is never handed back to the pool.
    owned: bool,
}

impl Buf {
    /// Uninitialized.
    pub fn new(len: usize) -> Buf {
        Buf { ptr: dev().alloc(len * 4), len, owned: true }
    }
    pub fn zeros(len: usize) -> Buf {
        let b = Buf::new(len);
        b.zero();
        b
    }
    pub fn zero(&self) {
        let d = dev();
        if self.len > 0 {
            d.check(unsafe { (d.api.memset_d32)(self.ptr, 0, self.len) }, "cuMemsetD32");
        }
    }
    pub fn from_f32(v: &[f32]) -> Buf {
        let b = Buf::new(v.len());
        b.upload(v.as_ptr() as *const c_void);
        b
    }
    pub fn from_i32(v: &[i32]) -> Buf {
        let b = Buf::new(v.len());
        b.upload(v.as_ptr() as *const c_void);
        b
    }
    fn upload(&self, src: *const c_void) {
        let d = dev();
        if self.len > 0 {
            d.check(unsafe { (d.api.memcpy_htod)(self.ptr, src, self.len * 4) }, "cuMemcpyHtoD");
        }
    }
    /// Waits for every kernel launched so far.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.len];
        let d = dev();
        if self.len > 0 {
            d.check(unsafe { (d.api.memcpy_dtoh)(v.as_mut_ptr() as *mut c_void, self.ptr, self.len * 4) }, "cuMemcpyDtoH");
        }
        v
    }
    pub fn arg(&self) -> Arg {
        Arg::P(self.ptr)
    }
    /// The slots [offset, offset + len) of this buffer, as a buffer of their own. The view must
    /// not be used after this buffer is dropped.
    pub fn view(&self, offset: usize, len: usize) -> Buf {
        assert!(offset + len <= self.len, "view [{offset}, {}) of a buffer of {}", offset + len, self.len);
        Buf { ptr: self.ptr + 4 * offset as u64, len, owned: false }
    }
    /// A buffer of doubles (two 4-byte slots each).
    pub fn from_f64(v: &[f64]) -> Buf {
        let b = Buf::new(2 * v.len());
        b.upload(v.as_ptr() as *const c_void);
        b
    }
    /// Waits for every kernel launched so far.
    pub fn to_f64(&self) -> Vec<f64> {
        let mut v = vec![0f64; self.len / 2];
        let d = dev();
        if self.len > 0 {
            d.check(unsafe { (d.api.memcpy_dtoh)(v.as_mut_ptr() as *mut c_void, self.ptr, self.len * 4) }, "cuMemcpyDtoH");
        }
        v
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        if let Some(Ok(d)) = DEVICE.get() {
            d.pool.lock().unwrap().entry(self.len * 4).or_default().push(self.ptr);
        }
    }
}
