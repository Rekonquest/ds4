// DS4 (DwarfStar) -- dynamic CUDA driver runtime.
//
// This module deliberately avoids link-time CUDA dependencies. It loads the
// NVIDIA driver and NVRTC at runtime, compiles a compact set of DS4 decode
// kernels to PTX, and JIT-loads that PTX through the driver API.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

use ds4_quant::luts::{IQ2XXS_GRID, KSIGNS_IQ2XS};
use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

pub type CuDevicePtr = u64;
type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type NvrtcResult = c_int;
type NvrtcProgram = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const NVRTC_SUCCESS: NvrtcResult = 0;
const CU_MEM_ATTACH_GLOBAL: c_uint = 1;

#[cfg(windows)]
type LibraryHandle = *mut c_void;

#[cfg(unix)]
type LibraryHandle = *mut c_void;

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> LibraryHandle;
    fn GetProcAddress(handle: LibraryHandle, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(handle: LibraryHandle) -> i32;
}

#[cfg(windows)]
fn load_library(path: &Path) -> LibraryHandle {
    use std::os::windows::ffi::OsStrExt;
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    unsafe { LoadLibraryW(wide.as_ptr()) }
}

#[cfg(windows)]
fn symbol(handle: LibraryHandle, name: &CStr) -> *mut c_void {
    unsafe { GetProcAddress(handle, name.as_ptr()) }
}

#[cfg(windows)]
fn free_library(handle: LibraryHandle) {
    unsafe {
        let _ = FreeLibrary(handle);
    }
}

#[cfg(unix)]
fn load_library(path: &Path) -> LibraryHandle {
    let Ok(name) = CString::new(path.to_string_lossy().as_bytes()) else {
        return ptr::null_mut();
    };
    unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) }
}

#[cfg(unix)]
fn symbol(handle: LibraryHandle, name: &CStr) -> *mut c_void {
    unsafe { libc::dlsym(handle, name.as_ptr()) }
}

#[cfg(unix)]
fn free_library(handle: LibraryHandle) {
    unsafe {
        let _ = libc::dlclose(handle);
    }
}

struct DynamicLibrary {
    handle: LibraryHandle,
    path: PathBuf,
}

impl DynamicLibrary {
    fn open_any(candidates: &[PathBuf], label: &str) -> Ds4Result<Self> {
        for candidate in candidates {
            let handle = load_library(candidate);
            if !handle.is_null() {
                return Ok(Self {
                    handle,
                    path: candidate.clone(),
                });
            }
        }
        Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            format!("{label} runtime library was not found"),
        ))
    }

    unsafe fn get<T: Copy>(&self, name: &str) -> Ds4Result<T> {
        let cname = CString::new(name).map_err(|e| {
            Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!("invalid symbol name {name:?}: {e}"),
            )
        })?;
        let raw = symbol(self.handle, cname.as_c_str());
        if raw.is_null() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                format!("{} missing symbol {name}", self.path.display()),
            ));
        }
        Ok(std::mem::transmute_copy(&raw))
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            free_library(self.handle);
        }
    }
}

unsafe impl Send for DynamicLibrary {}
unsafe impl Sync for DynamicLibrary {}

type CuInit = unsafe extern "system" fn(c_uint) -> CuResult;
type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, c_int) -> CuResult;
type CuCtxCreate = unsafe extern "system" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
type CuCtxSetCurrent = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxDestroy = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxSynchronize = unsafe extern "system" fn() -> CuResult;
type CuMemAlloc = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemAllocManaged = unsafe extern "system" fn(*mut CuDevicePtr, usize, c_uint) -> CuResult;
type CuMemFree = unsafe extern "system" fn(CuDevicePtr) -> CuResult;
type CuMemcpyHtoD = unsafe extern "system" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type CuMemcpyDtoH = unsafe extern "system" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type CuModuleLoadData = unsafe extern "system" fn(*mut CuModule, *const c_void) -> CuResult;
type CuModuleUnload = unsafe extern "system" fn(CuModule) -> CuResult;
type CuModuleGetFunction =
    unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuLaunchKernel = unsafe extern "system" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type CuGetErrorString = unsafe extern "system" fn(CuResult, *mut *const c_char) -> CuResult;

struct DriverApi {
    _lib: DynamicLibrary,
    cu_init: CuInit,
    cu_device_get: CuDeviceGet,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_set_current: CuCtxSetCurrent,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_synchronize: CuCtxSynchronize,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_alloc_managed: CuMemAllocManaged,
    cu_mem_free: CuMemFree,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_module_load_data: CuModuleLoadData,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_launch_kernel: CuLaunchKernel,
    cu_get_error_string: CuGetErrorString,
}

impl DriverApi {
    fn load() -> Ds4Result<Self> {
        let lib = DynamicLibrary::open_any(&driver_candidates(), "CUDA driver")?;
        unsafe {
            Ok(Self {
                cu_init: lib.get("cuInit")?,
                cu_device_get: lib.get("cuDeviceGet")?,
                cu_ctx_create: lib.get("cuCtxCreate_v2")?,
                cu_ctx_set_current: lib.get("cuCtxSetCurrent")?,
                cu_ctx_destroy: lib.get("cuCtxDestroy_v2")?,
                cu_ctx_synchronize: lib.get("cuCtxSynchronize")?,
                cu_mem_alloc: lib.get("cuMemAlloc_v2")?,
                cu_mem_alloc_managed: lib.get("cuMemAllocManaged")?,
                cu_mem_free: lib.get("cuMemFree_v2")?,
                cu_memcpy_htod: lib.get("cuMemcpyHtoD_v2")?,
                cu_memcpy_dtoh: lib.get("cuMemcpyDtoH_v2")?,
                cu_module_load_data: lib.get("cuModuleLoadData")?,
                cu_module_unload: lib.get("cuModuleUnload")?,
                cu_module_get_function: lib.get("cuModuleGetFunction")?,
                cu_launch_kernel: lib.get("cuLaunchKernel")?,
                cu_get_error_string: lib.get("cuGetErrorString")?,
                _lib: lib,
            })
        }
    }

    fn check(&self, result: CuResult, op: &str) -> Ds4Result<()> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let mut raw: *const c_char = ptr::null();
        let message = unsafe {
            if (self.cu_get_error_string)(result, &mut raw) == CUDA_SUCCESS && !raw.is_null() {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            } else {
                format!("CUDA error {result}")
            }
        };
        Err(Ds4Error::new(
            Ds4ErrorKind::Other,
            format!("{op}: {message}"),
        ))
    }
}

unsafe impl Send for DriverApi {}
unsafe impl Sync for DriverApi {}

type NvrtcCreateProgram = unsafe extern "system" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;
type NvrtcCompileProgram =
    unsafe extern "system" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
type NvrtcGetPtxSize = unsafe extern "system" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
type NvrtcGetPtx = unsafe extern "system" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
type NvrtcGetProgramLogSize = unsafe extern "system" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
type NvrtcGetProgramLog = unsafe extern "system" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
type NvrtcDestroyProgram = unsafe extern "system" fn(*mut NvrtcProgram) -> NvrtcResult;
type NvrtcGetErrorString = unsafe extern "system" fn(NvrtcResult) -> *const c_char;

struct NvrtcApi {
    _lib: DynamicLibrary,
    create_program: NvrtcCreateProgram,
    compile_program: NvrtcCompileProgram,
    get_ptx_size: NvrtcGetPtxSize,
    get_ptx: NvrtcGetPtx,
    get_program_log_size: NvrtcGetProgramLogSize,
    get_program_log: NvrtcGetProgramLog,
    destroy_program: NvrtcDestroyProgram,
    get_error_string: NvrtcGetErrorString,
}

impl NvrtcApi {
    fn load() -> Ds4Result<Self> {
        let lib = DynamicLibrary::open_any(&nvrtc_candidates(), "NVRTC")?;
        unsafe {
            Ok(Self {
                create_program: lib.get("nvrtcCreateProgram")?,
                compile_program: lib.get("nvrtcCompileProgram")?,
                get_ptx_size: lib.get("nvrtcGetPTXSize")?,
                get_ptx: lib.get("nvrtcGetPTX")?,
                get_program_log_size: lib.get("nvrtcGetProgramLogSize")?,
                get_program_log: lib.get("nvrtcGetProgramLog")?,
                destroy_program: lib.get("nvrtcDestroyProgram")?,
                get_error_string: lib.get("nvrtcGetErrorString")?,
                _lib: lib,
            })
        }
    }

    fn error_message(&self, result: NvrtcResult) -> String {
        let ptr = unsafe { (self.get_error_string)(result) };
        if ptr.is_null() {
            format!("NVRTC error {result}")
        } else {
            unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
        }
    }
}

unsafe impl Send for NvrtcApi {}
unsafe impl Sync for NvrtcApi {}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeKernels {
    pub embedding: CuFunction,
    pub rmsnorm: CuFunction,
    pub matvec: CuFunction,
    pub add: CuFunction,
    pub silu_product: CuFunction,
    pub rope: CuFunction,
    pub store_cache: CuFunction,
    pub attention: CuFunction,
}

pub struct CudaRuntime {
    driver: DriverApi,
    context: CuContext,
    module: CuModule,
    kernels: RuntimeKernels,
}

unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

impl CudaRuntime {
    pub fn load() -> Ds4Result<Arc<Self>> {
        let driver = DriverApi::load()?;
        unsafe {
            driver.check((driver.cu_init)(0), "cuInit")?;
            let mut device = 0;
            driver.check((driver.cu_device_get)(&mut device, 0), "cuDeviceGet")?;
            let mut context = ptr::null_mut();
            driver.check(
                (driver.cu_ctx_create)(&mut context, 0, device),
                "cuCtxCreate",
            )?;

            let nvrtc = NvrtcApi::load()?;
            let source = runtime_kernel_source();
            let ptx = compile_ptx(&nvrtc, &source)?;
            let mut module = ptr::null_mut();
            driver.check(
                (driver.cu_module_load_data)(&mut module, ptx.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
            let kernels = RuntimeKernels {
                embedding: get_function(&driver, module, "ds4_embedding")?,
                rmsnorm: get_function(&driver, module, "ds4_rmsnorm")?,
                matvec: get_function(&driver, module, "ds4_matvec")?,
                add: get_function(&driver, module, "ds4_add")?,
                silu_product: get_function(&driver, module, "ds4_silu_product")?,
                rope: get_function(&driver, module, "ds4_rope")?,
                store_cache: get_function(&driver, module, "ds4_store_cache")?,
                attention: get_function(&driver, module, "ds4_attention_decode")?,
            };
            Ok(Arc::new(Self {
                driver,
                context,
                module,
                kernels,
            }))
        }
    }

    pub fn kernels(&self) -> RuntimeKernels {
        self.kernels
    }

    fn ensure_current(&self) -> Ds4Result<()> {
        unsafe {
            self.driver.check(
                (self.driver.cu_ctx_set_current)(self.context),
                "cuCtxSetCurrent",
            )
        }
    }

    pub fn alloc_device(runtime: &Arc<Self>, bytes: usize) -> Ds4Result<DeviceMem> {
        runtime.ensure_current()?;
        let mut ptr = 0;
        unsafe {
            runtime
                .driver
                .check((runtime.driver.cu_mem_alloc)(&mut ptr, bytes), "cuMemAlloc")?;
        }
        Ok(DeviceMem {
            ptr,
            bytes,
            runtime: Arc::clone(runtime),
        })
    }

    pub fn alloc_managed(runtime: &Arc<Self>, bytes: usize) -> Ds4Result<DeviceMem> {
        runtime.ensure_current()?;
        let mut ptr = 0;
        unsafe {
            runtime.driver.check(
                (runtime.driver.cu_mem_alloc_managed)(&mut ptr, bytes, CU_MEM_ATTACH_GLOBAL),
                "cuMemAllocManaged",
            )?;
        }
        Ok(DeviceMem {
            ptr,
            bytes,
            runtime: Arc::clone(runtime),
        })
    }

    pub fn copy_htod(&self, dst: CuDevicePtr, src: &[u8]) -> Ds4Result<()> {
        self.ensure_current()?;
        unsafe {
            self.driver.check(
                (self.driver.cu_memcpy_htod)(dst, src.as_ptr().cast(), src.len()),
                "cuMemcpyHtoD",
            )
        }
    }

    pub fn copy_dtoh(&self, dst: &mut [u8], src: CuDevicePtr) -> Ds4Result<()> {
        self.ensure_current()?;
        unsafe {
            self.driver.check(
                (self.driver.cu_memcpy_dtoh)(dst.as_mut_ptr().cast(), src, dst.len()),
                "cuMemcpyDtoH",
            )
        }
    }

    pub fn synchronize(&self) -> Ds4Result<()> {
        self.ensure_current()?;
        unsafe {
            self.driver
                .check((self.driver.cu_ctx_synchronize)(), "cuCtxSynchronize")
        }
    }

    pub(crate) fn launch(
        &self,
        function: CuFunction,
        grid_x: u32,
        block_x: u32,
        params: &mut [*mut c_void],
    ) -> Ds4Result<()> {
        self.ensure_current()?;
        unsafe {
            self.driver.check(
                (self.driver.cu_launch_kernel)(
                    function,
                    grid_x,
                    1,
                    1,
                    block_x,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    fn free(&self, ptr: CuDevicePtr) {
        if ptr == 0 {
            return;
        }
        let _ = self.ensure_current();
        unsafe {
            let _ = (self.driver.cu_mem_free)(ptr);
        }
    }
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.context);
            if !self.module.is_null() {
                let _ = (self.driver.cu_module_unload)(self.module);
            }
            if !self.context.is_null() {
                let _ = (self.driver.cu_ctx_destroy)(self.context);
            }
        }
    }
}

pub struct DeviceMem {
    ptr: CuDevicePtr,
    bytes: usize,
    runtime: Arc<CudaRuntime>,
}

unsafe impl Send for DeviceMem {}
unsafe impl Sync for DeviceMem {}

impl DeviceMem {
    pub fn ptr(&self) -> CuDevicePtr {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn copy_from(&self, data: &[u8]) -> Ds4Result<()> {
        if data.len() > self.bytes {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "host copy {} exceeds device allocation {}",
                    data.len(),
                    self.bytes
                ),
            ));
        }
        self.runtime.copy_htod(self.ptr, data)
    }

    pub fn copy_to(&self, data: &mut [u8]) -> Ds4Result<()> {
        if data.len() > self.bytes {
            return Err(Ds4Error::new(
                Ds4ErrorKind::InvalidArgument,
                format!(
                    "host destination {} exceeds device allocation {}",
                    data.len(),
                    self.bytes
                ),
            ));
        }
        self.runtime.copy_dtoh(data, self.ptr)
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        self.runtime.free(self.ptr);
    }
}

fn get_function(driver: &DriverApi, module: CuModule, name: &str) -> Ds4Result<CuFunction> {
    let cname = CString::new(name).map_err(|e| {
        Ds4Error::new(
            Ds4ErrorKind::InvalidArgument,
            format!("invalid kernel name {name:?}: {e}"),
        )
    })?;
    let mut function = ptr::null_mut();
    unsafe {
        driver.check(
            (driver.cu_module_get_function)(&mut function, module, cname.as_ptr()),
            "cuModuleGetFunction",
        )?;
    }
    Ok(function)
}

fn compile_ptx(nvrtc: &NvrtcApi, source: &str) -> Ds4Result<Vec<u8>> {
    let src = CString::new(source).map_err(|e| {
        Ds4Error::new(
            Ds4ErrorKind::InvalidArgument,
            format!("CUDA source contains interior nul: {e}"),
        )
    })?;
    let name = CString::new("ds4_runtime.cu").expect("static string");
    let mut program = ptr::null_mut();
    unsafe {
        let create = (nvrtc.create_program)(
            &mut program,
            src.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        );
        if create != NVRTC_SUCCESS {
            return Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                format!("nvrtcCreateProgram: {}", nvrtc.error_message(create)),
            ));
        }
    }

    let options = [
        CString::new("--gpu-architecture=compute_80").expect("static string"),
        CString::new("--use_fast_math").expect("static string"),
    ];
    let option_ptrs: Vec<*const c_char> = options.iter().map(|s| s.as_ptr()).collect();
    let compile = unsafe {
        (nvrtc.compile_program)(program, option_ptrs.len() as c_int, option_ptrs.as_ptr())
    };
    if compile != NVRTC_SUCCESS {
        let log = nvrtc_log(nvrtc, program);
        unsafe {
            let _ = (nvrtc.destroy_program)(&mut program);
        }
        return Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            format!(
                "nvrtcCompileProgram: {}; {log}",
                nvrtc.error_message(compile)
            ),
        ));
    }

    let mut size = 0usize;
    unsafe {
        let res = (nvrtc.get_ptx_size)(program, &mut size);
        if res != NVRTC_SUCCESS {
            let _ = (nvrtc.destroy_program)(&mut program);
            return Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                format!("nvrtcGetPTXSize: {}", nvrtc.error_message(res)),
            ));
        }
        let mut ptx = vec![0u8; size];
        let res = (nvrtc.get_ptx)(program, ptx.as_mut_ptr().cast());
        let _ = (nvrtc.destroy_program)(&mut program);
        if res != NVRTC_SUCCESS {
            return Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                format!("nvrtcGetPTX: {}", nvrtc.error_message(res)),
            ));
        }
        Ok(ptx)
    }
}

fn nvrtc_log(nvrtc: &NvrtcApi, program: NvrtcProgram) -> String {
    let mut size = 0usize;
    unsafe {
        if (nvrtc.get_program_log_size)(program, &mut size) != NVRTC_SUCCESS || size == 0 {
            return String::new();
        }
        let mut log = vec![0u8; size];
        if (nvrtc.get_program_log)(program, log.as_mut_ptr().cast()) != NVRTC_SUCCESS {
            return String::new();
        }
        String::from_utf8_lossy(&log)
            .trim_matches(char::from(0))
            .to_string()
    }
}

fn driver_candidates() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        vec![PathBuf::from("nvcuda.dll")]
    }
    #[cfg(unix)]
    {
        vec![PathBuf::from("libcuda.so.1"), PathBuf::from("libcuda.so")]
    }
}

fn nvrtc_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(path) = std::env::var("DS4_NVRTC_DLL") {
        out.push(PathBuf::from(path));
    }
    if let Ok(path) = std::env::var("NVRTC_DLL") {
        out.push(PathBuf::from(path));
    }
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        collect_nvrtc_from_bin(&mut out, PathBuf::from(cuda_path).join("bin"));
    }
    #[cfg(windows)]
    {
        out.push(PathBuf::from("nvrtc64_130_0.dll"));
        out.push(PathBuf::from("nvrtc64_120_0.dll"));
        out.push(PathBuf::from("nvrtc64_112_0.dll"));
        let cuda_root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if let Ok(entries) = std::fs::read_dir(cuda_root) {
            for entry in entries.flatten() {
                collect_nvrtc_from_bin(&mut out, entry.path().join("bin"));
            }
        }
        out.push(PathBuf::from(
            r"C:\Program Files\Blackmagic Design\DaVinci Resolve\nvrtc64_120_0.dll",
        ));
    }
    #[cfg(unix)]
    {
        out.push(PathBuf::from("libnvrtc.so"));
        out.push(PathBuf::from("libnvrtc.so.12"));
        out.push(PathBuf::from("/usr/local/cuda/lib64/libnvrtc.so"));
    }
    dedup_paths(out)
}

fn collect_nvrtc_from_bin(out: &mut Vec<PathBuf>, bin: PathBuf) {
    let Ok(entries) = std::fs::read_dir(bin) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if name.starts_with("nvrtc64_") && name.ends_with(".dll") {
            out.push(path);
        }
    }
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashMap::<String, ()>::new();
    let mut out = Vec::new();
    for path in paths {
        let key = path.to_string_lossy().to_lowercase();
        if seen.insert(key, ()).is_none() {
            out.push(path);
        }
    }
    out
}

pub fn runtime_kernel_source() -> String {
    let mut src = String::new();
    src.push_str("__constant__ unsigned char DS4_KSIGNS[128] = {");
    for (idx, value) in KSIGNS_IQ2XS.iter().enumerate() {
        if idx != 0 {
            src.push(',');
        }
        src.push_str(&value.to_string());
    }
    src.push_str("};\n__constant__ unsigned long long DS4_IQ2_GRID[256] = {");
    for (idx, value) in IQ2XXS_GRID.iter().enumerate() {
        if idx != 0 {
            src.push(',');
        }
        src.push_str(&format!("0x{value:016x}ULL"));
    }
    src.push_str("};\n");
    src.push_str(RUNTIME_KERNEL_BODY);
    src
}

const RUNTIME_KERNEL_BODY: &str = r#"
#define DS4_DTYPE_F32 0
#define DS4_DTYPE_F16 1
#define DS4_DTYPE_Q8_0 2
#define DS4_DTYPE_Q4_K 3
#define DS4_DTYPE_Q3_K 4
#define DS4_DTYPE_Q2_K 5
#define DS4_DTYPE_IQ2_XXS 6

__device__ unsigned short ds4_u16(const unsigned char * p) {
    return ((unsigned short)p[0]) | ((unsigned short)p[1] << 8);
}

__device__ unsigned int ds4_u32(const unsigned char * p) {
    return ((unsigned int)p[0]) | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

__device__ float ds4_f32(const unsigned char * p) {
    return __uint_as_float(ds4_u32(p));
}

__device__ float ds4_f16(unsigned short h) {
    unsigned int sign = ((unsigned int)h & 0x8000U) << 16;
    int exp = (int)((h >> 10) & 0x1fU);
    unsigned int mant = (unsigned int)h & 0x03ffU;
    if (exp == 0) {
        if (mant == 0) {
            return __uint_as_float(sign);
        }
        while ((mant & 0x0400U) == 0) {
            mant <<= 1;
            exp -= 1;
        }
        exp += 1;
        mant &= ~0x0400U;
    } else if (exp == 31) {
        return __uint_as_float(sign | 0x7f800000U | (mant << 13));
    }
    exp = exp + (127 - 15);
    return __uint_as_float(sign | ((unsigned int)exp << 23) | (mant << 13));
}

__device__ void ds4_scale_min_k4(int j, const unsigned char * q, int * d, int * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ int ds4_q3_scale(const unsigned char * scales, int j) {
    int low = (j < 8) ? (scales[j] & 0x0f) : ((scales[j - 8] >> 4) & 0x0f);
    int high = (scales[8 + (j & 3)] >> (2 * (j >> 2))) & 3;
    return (low | (high << 4)) - 32;
}

__device__ float ds4_load_q8_0(const unsigned char * data, unsigned long long index) {
    unsigned long long block = index >> 5;
    int r = (int)(index & 31ULL);
    const unsigned char * b = data + block * 34ULL;
    float d = ds4_f16(ds4_u16(b));
    signed char q = (signed char)b[2 + r];
    return d * (float)q;
}

__device__ float ds4_load_q4_k(const unsigned char * data, unsigned long long index) {
    unsigned long long block = index >> 8;
    int r = (int)(index & 255ULL);
    const unsigned char * b = data + block * 144ULL;
    float d = ds4_f16(ds4_u16(b));
    float dmin = ds4_f16(ds4_u16(b + 2));
    const unsigned char * scales = b + 4;
    const unsigned char * qs = b + 16;
    int group = r >> 6;
    int local = r & 63;
    int sc = 0;
    int mn = 0;
    ds4_scale_min_k4(group * 2 + (local >= 32), scales, &sc, &mn);
    unsigned char packed = qs[group * 32 + (local & 31)];
    int q = (local < 32) ? (packed & 0x0f) : (packed >> 4);
    return d * (float)sc * (float)q - dmin * (float)mn;
}

__device__ float ds4_load_q3_k(const unsigned char * data, unsigned long long index) {
    unsigned long long block = index >> 8;
    int r = (int)(index & 255ULL);
    const unsigned char * b = data + block * 110ULL;
    const unsigned char * hmask = b;
    const unsigned char * qs = b + 32;
    const unsigned char * scales = b + 96;
    float d = ds4_f16(ds4_u16(b + 108));
    int half = r >> 7;
    int rem = r & 127;
    int j = rem >> 5;
    int within = rem & 31;
    int scale_idx = half * 8 + j * 2 + (within >= 16);
    int q_off = half * 32 + within;
    int shift = j * 2;
    int q = (qs[q_off] >> shift) & 3;
    int mask = 1 << (half * 4 + j);
    int bias = (hmask[within] & mask) ? 0 : 4;
    return d * (float)ds4_q3_scale(scales, scale_idx) * (float)(q - bias);
}

__device__ float ds4_load_q2_k(const unsigned char * data, unsigned long long index) {
    unsigned long long block = index >> 8;
    int r = (int)(index & 255ULL);
    const unsigned char * b = data + block * 84ULL;
    const unsigned char * scales = b;
    const unsigned char * qs = b + 16;
    float d = ds4_f16(ds4_u16(b + 80));
    float dmin = ds4_f16(ds4_u16(b + 82));
    int half = r >> 7;
    int rem = r & 127;
    int pair = rem >> 5;
    int within = rem & 31;
    int scale_idx = half * 8 + pair * 2 + (within >= 16);
    int shift = pair * 2;
    int q = (qs[half * 32 + within] >> shift) & 3;
    int sc = scales[scale_idx];
    return d * (float)(sc & 0x0f) * (float)q - dmin * (float)(sc >> 4);
}

__device__ float ds4_load_iq2_xxs(const unsigned char * data, unsigned long long index) {
    unsigned long long block = index >> 8;
    int r = (int)(index & 255ULL);
    const unsigned char * b = data + block * 66ULL;
    float d = ds4_f16(ds4_u16(b));
    const unsigned char * q = b + 2;
    int ib32 = r >> 5;
    int sub = (r >> 3) & 3;
    int lane = r & 7;
    unsigned int q0 = (unsigned int)ds4_u16(q + (ib32 * 4 + 0) * 2);
    unsigned int q1 = (unsigned int)ds4_u16(q + (ib32 * 4 + 1) * 2);
    unsigned int q2 = (unsigned int)ds4_u16(q + (ib32 * 4 + 2) * 2);
    unsigned int q3 = (unsigned int)ds4_u16(q + (ib32 * 4 + 3) * 2);
    unsigned int aux0 = q0 | (q1 << 16);
    unsigned int aux1 = q2 | (q3 << 16);
    unsigned int grid_idx = (sub == 0) ? (aux0 & 0xffU) :
                            (sub == 1) ? ((aux0 >> 8) & 0xffU) :
                            (sub == 2) ? ((aux0 >> 16) & 0xffU) :
                                         ((aux0 >> 24) & 0xffU);
    unsigned int signs = (unsigned int)DS4_KSIGNS[(aux1 >> (7 * sub)) & 0x7fU];
    unsigned int grid_byte = (unsigned int)((DS4_IQ2_GRID[grid_idx] >> (8 * lane)) & 0xffULL);
    float sign = (signs & (1U << lane)) ? -1.0f : 1.0f;
    float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
    return db * (float)grid_byte * sign;
}

__device__ float ds4_load_weight(const unsigned char * data, int dtype, unsigned long long index) {
    if (dtype == DS4_DTYPE_F32) {
        return ds4_f32(data + index * 4ULL);
    }
    if (dtype == DS4_DTYPE_F16) {
        return ds4_f16(ds4_u16(data + index * 2ULL));
    }
    if (dtype == DS4_DTYPE_Q8_0) {
        return ds4_load_q8_0(data, index);
    }
    if (dtype == DS4_DTYPE_Q4_K) {
        return ds4_load_q4_k(data, index);
    }
    if (dtype == DS4_DTYPE_Q3_K) {
        return ds4_load_q3_k(data, index);
    }
    if (dtype == DS4_DTYPE_Q2_K) {
        return ds4_load_q2_k(data, index);
    }
    if (dtype == DS4_DTYPE_IQ2_XXS) {
        return ds4_load_iq2_xxs(data, index);
    }
    return 0.0f;
}

extern "C" __global__ void ds4_embedding(
    const unsigned char * weights,
    int dtype,
    int token,
    int hidden,
    float * out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    out[i] = ds4_load_weight(weights, dtype, (unsigned long long)token * (unsigned long long)hidden + (unsigned long long)i);
}

extern "C" __global__ void ds4_rmsnorm(
    const float * x,
    const unsigned char * weight,
    int weight_dtype,
    float * out,
    int n,
    float eps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    float scale = rsqrtf(sum / (float)n + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] * scale * ds4_load_weight(weight, weight_dtype, (unsigned long long)i);
    }
}

extern "C" __global__ void ds4_matvec(
    const float * input,
    const unsigned char * weights,
    int weight_dtype,
    float * out,
    int input_dim,
    int out_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;
    float sum = 0.0f;
    for (int i = 0; i < input_dim; ++i) {
        unsigned long long index = (unsigned long long)i * (unsigned long long)out_dim + (unsigned long long)j;
        sum += input[i] * ds4_load_weight(weights, weight_dtype, index);
    }
    out[j] = sum;
}

extern "C" __global__ void ds4_add(float * dst, const float * src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i];
    }
}

extern "C" __global__ void ds4_silu_product(float * gate, const float * up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

extern "C" __global__ void ds4_rope(
    float * x,
    int pos,
    int n_heads,
    int head_dim,
    float freq_base) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim >> 1;
    int total = n_heads * half;
    if (pair >= total) return;
    int head = pair / half;
    int i = pair - head * half;
    int off = head * head_dim + i * 2;
    float exponent = 2.0f * (float)i / (float)head_dim;
    float freq = expf(exponent * logf(freq_base));
    float theta = (float)pos / freq;
    float s = sinf(theta);
    float c = cosf(theta);
    float a = x[off];
    float b = x[off + 1];
    x[off] = a * c - b * s;
    x[off + 1] = a * s + b * c;
}

extern "C" __global__ void ds4_store_cache(
    const float * src,
    float * cache,
    int cache_pos,
    int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        cache[cache_pos * hidden + i] = src[i];
    }
}

extern "C" __global__ void ds4_attention_decode(
    const float * q,
    const float * k_cache,
    const float * v_cache,
    float * out,
    int seq_len,
    int n_heads,
    int head_dim) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads) return;
    float scale = rsqrtf((float)head_dim);
    float max_score = -3.402823466e+38f;
    for (int t = 0; t < seq_len; ++t) {
        float score = 0.0f;
        int base = t * n_heads * head_dim + h * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            score += q[h * head_dim + d] * k_cache[base + d];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }
    float denom = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        float score = 0.0f;
        int base = t * n_heads * head_dim + h * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            score += q[h * head_dim + d] * k_cache[base + d];
        }
        denom += expf(score * scale - max_score);
    }
    for (int d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            float score = 0.0f;
            int base = t * n_heads * head_dim + h * head_dim;
            for (int k = 0; k < head_dim; ++k) {
                score += q[h * head_dim + k] * k_cache[base + k];
            }
            float w = expf(score * scale - max_score) / denom;
            acc += w * v_cache[base + d];
        }
        out[h * head_dim + d] = acc;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_runtime_source_contains_iq2_tables() {
        let source = runtime_kernel_source();
        assert!(source.contains("DS4_KSIGNS[128]"));
        assert!(source.contains("DS4_IQ2_GRID[256]"));
        assert!(source.contains("ds4_matvec"));
        assert!(source.contains("ds4_load_q3_k"));
    }

    #[test]
    fn nvrtc_candidate_list_is_non_empty() {
        assert!(!nvrtc_candidates().is_empty());
    }
}
