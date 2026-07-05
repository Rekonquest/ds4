#[cfg(windows)]
mod imp {
    use std::ffi::{c_char, c_void, CString};
    use std::mem::size_of;
    use std::ptr::{null, null_mut};
    use std::sync::Arc;

    use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

    use crate::kernels::ARC_KERNEL_SRC;

    type ClInt = i32;
    type ClUint = u32;
    type ClUlong = u64;
    type ClBool = ClUint;
    type ClBitfield = ClUlong;
    type ClPlatformId = *mut c_void;
    type ClDeviceId = *mut c_void;
    type ClContext = *mut c_void;
    type ClCommandQueue = *mut c_void;
    type ClProgram = *mut c_void;
    pub(crate) type ClKernel = *mut c_void;
    type ClMem = *mut c_void;
    type ClContextProperties = isize;

    const CL_SUCCESS: ClInt = 0;
    const CL_TRUE: ClBool = 1;
    const CL_DEVICE_TYPE_GPU: ClBitfield = 1 << 2;
    const CL_MEM_READ_WRITE: ClBitfield = 1;
    const CL_MEM_COPY_HOST_PTR: ClBitfield = 1 << 5;
    const CL_PLATFORM_NAME: ClUint = 0x0902;
    const CL_PLATFORM_VENDOR: ClUint = 0x0903;
    const CL_DEVICE_NAME: ClUint = 0x102B;
    const CL_DEVICE_VENDOR: ClUint = 0x102C;
    const CL_PROGRAM_BUILD_LOG: ClUint = 0x1183;

    type ClGetPlatformIDs =
        unsafe extern "system" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt;
    type ClGetPlatformInfo =
        unsafe extern "system" fn(ClPlatformId, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
    type ClGetDeviceIDs = unsafe extern "system" fn(
        ClPlatformId,
        ClBitfield,
        ClUint,
        *mut ClDeviceId,
        *mut ClUint,
    ) -> ClInt;
    type ClGetDeviceInfo =
        unsafe extern "system" fn(ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
    type ClCreateContext = unsafe extern "system" fn(
        *const ClContextProperties,
        ClUint,
        *const ClDeviceId,
        *mut c_void,
        *mut c_void,
        *mut ClInt,
    ) -> ClContext;
    type ClCreateCommandQueue =
        unsafe extern "system" fn(ClContext, ClDeviceId, ClBitfield, *mut ClInt) -> ClCommandQueue;
    type ClReleaseContext = unsafe extern "system" fn(ClContext) -> ClInt;
    type ClReleaseCommandQueue = unsafe extern "system" fn(ClCommandQueue) -> ClInt;
    type ClCreateProgramWithSource = unsafe extern "system" fn(
        ClContext,
        ClUint,
        *const *const c_char,
        *const usize,
        *mut ClInt,
    ) -> ClProgram;
    type ClBuildProgram = unsafe extern "system" fn(
        ClProgram,
        ClUint,
        *const ClDeviceId,
        *const c_char,
        *mut c_void,
        *mut c_void,
    ) -> ClInt;
    type ClGetProgramBuildInfo = unsafe extern "system" fn(
        ClProgram,
        ClDeviceId,
        ClUint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> ClInt;
    type ClReleaseProgram = unsafe extern "system" fn(ClProgram) -> ClInt;
    type ClCreateKernel =
        unsafe extern "system" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel;
    type ClReleaseKernel = unsafe extern "system" fn(ClKernel) -> ClInt;
    type ClCreateBuffer =
        unsafe extern "system" fn(ClContext, ClBitfield, usize, *mut c_void, *mut ClInt) -> ClMem;
    type ClReleaseMemObject = unsafe extern "system" fn(ClMem) -> ClInt;
    type ClSetKernelArg =
        unsafe extern "system" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt;
    type ClEnqueueNDRangeKernel = unsafe extern "system" fn(
        ClCommandQueue,
        ClKernel,
        ClUint,
        *const usize,
        *const usize,
        *const usize,
        ClUint,
        *const c_void,
        *mut c_void,
    ) -> ClInt;
    type ClFinish = unsafe extern "system" fn(ClCommandQueue) -> ClInt;
    type ClEnqueueReadBuffer = unsafe extern "system" fn(
        ClCommandQueue,
        ClMem,
        ClBool,
        usize,
        usize,
        *mut c_void,
        ClUint,
        *const c_void,
        *mut c_void,
    ) -> ClInt;
    type ClEnqueueWriteBuffer = unsafe extern "system" fn(
        ClCommandQueue,
        ClMem,
        ClBool,
        usize,
        usize,
        *const c_void,
        ClUint,
        *const c_void,
        *mut c_void,
    ) -> ClInt;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    struct DynamicLibrary {
        handle: *mut c_void,
    }

    unsafe impl Send for DynamicLibrary {}
    unsafe impl Sync for DynamicLibrary {}

    impl DynamicLibrary {
        fn open(name: &str) -> Ds4Result<Self> {
            let mut wide = name.encode_utf16().collect::<Vec<u16>>();
            wide.push(0);
            let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
            if handle.is_null() {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::NotImplemented,
                    format!("{name} could not be loaded"),
                ));
            }
            Ok(Self { handle })
        }

        fn get<T: Copy>(&self, name: &str) -> Ds4Result<T> {
            let c_name = CString::new(name).expect("OpenCL symbol has no interior NUL");
            let ptr = unsafe { GetProcAddress(self.handle, c_name.as_ptr()) };
            if ptr.is_null() {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::NotImplemented,
                    format!("OpenCL symbol {name} is unavailable"),
                ));
            }
            Ok(unsafe { std::mem::transmute_copy(&ptr) })
        }
    }

    impl Drop for DynamicLibrary {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    FreeLibrary(self.handle);
                }
            }
        }
    }

    struct OpenCl {
        _library: DynamicLibrary,
        cl_get_platform_ids: ClGetPlatformIDs,
        cl_get_platform_info: ClGetPlatformInfo,
        cl_get_device_ids: ClGetDeviceIDs,
        cl_get_device_info: ClGetDeviceInfo,
        cl_create_context: ClCreateContext,
        cl_create_command_queue: ClCreateCommandQueue,
        cl_release_context: ClReleaseContext,
        cl_release_command_queue: ClReleaseCommandQueue,
        cl_create_program_with_source: ClCreateProgramWithSource,
        cl_build_program: ClBuildProgram,
        cl_get_program_build_info: ClGetProgramBuildInfo,
        cl_release_program: ClReleaseProgram,
        cl_create_kernel: ClCreateKernel,
        cl_release_kernel: ClReleaseKernel,
        cl_create_buffer: ClCreateBuffer,
        cl_release_mem_object: ClReleaseMemObject,
        cl_set_kernel_arg: ClSetKernelArg,
        cl_enqueue_nd_range_kernel: ClEnqueueNDRangeKernel,
        cl_finish: ClFinish,
        cl_enqueue_read_buffer: ClEnqueueReadBuffer,
        cl_enqueue_write_buffer: ClEnqueueWriteBuffer,
    }

    impl OpenCl {
        fn load() -> Ds4Result<Self> {
            let library = DynamicLibrary::open("OpenCL.dll")?;
            Ok(Self {
                cl_get_platform_ids: library.get("clGetPlatformIDs")?,
                cl_get_platform_info: library.get("clGetPlatformInfo")?,
                cl_get_device_ids: library.get("clGetDeviceIDs")?,
                cl_get_device_info: library.get("clGetDeviceInfo")?,
                cl_create_context: library.get("clCreateContext")?,
                cl_create_command_queue: library.get("clCreateCommandQueue")?,
                cl_release_context: library.get("clReleaseContext")?,
                cl_release_command_queue: library.get("clReleaseCommandQueue")?,
                cl_create_program_with_source: library.get("clCreateProgramWithSource")?,
                cl_build_program: library.get("clBuildProgram")?,
                cl_get_program_build_info: library.get("clGetProgramBuildInfo")?,
                cl_release_program: library.get("clReleaseProgram")?,
                cl_create_kernel: library.get("clCreateKernel")?,
                cl_release_kernel: library.get("clReleaseKernel")?,
                cl_create_buffer: library.get("clCreateBuffer")?,
                cl_release_mem_object: library.get("clReleaseMemObject")?,
                cl_set_kernel_arg: library.get("clSetKernelArg")?,
                cl_enqueue_nd_range_kernel: library.get("clEnqueueNDRangeKernel")?,
                cl_finish: library.get("clFinish")?,
                cl_enqueue_read_buffer: library.get("clEnqueueReadBuffer")?,
                cl_enqueue_write_buffer: library.get("clEnqueueWriteBuffer")?,
                _library: library,
            })
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum KernelId {
        EmbeddingWeight,
        RmsNormWeight,
        MatvecWeight,
        EmbeddingF32,
        RmsNormF32,
        MatvecF32,
        AddInplaceF32,
        AddScaledInplaceF32,
        SiluProductF32,
        RopeF32,
        StoreCacheF32,
        AttentionDecodeF32,
    }

    #[derive(Debug)]
    struct Kernels {
        embedding_weight: ClKernel,
        rmsnorm_weight: ClKernel,
        matvec_weight: ClKernel,
        embedding_f32: ClKernel,
        rmsnorm_f32: ClKernel,
        matvec_f32: ClKernel,
        add_inplace_f32: ClKernel,
        add_scaled_inplace_f32: ClKernel,
        silu_product_f32: ClKernel,
        rope_f32: ClKernel,
        store_cache_f32: ClKernel,
        attention_decode_f32: ClKernel,
    }

    pub struct ArcRuntime {
        opencl: OpenCl,
        context: ClContext,
        queue: ClCommandQueue,
        program: ClProgram,
        kernels: Kernels,
        platform_name: String,
        device_name: String,
    }

    unsafe impl Send for ArcRuntime {}
    unsafe impl Sync for ArcRuntime {}

    impl ArcRuntime {
        pub fn load() -> Ds4Result<Arc<Self>> {
            let opencl = OpenCl::load()?;
            let (platform, device, platform_name, device_name) = find_arc_device(&opencl)?;
            let _ = platform;
            let mut err = CL_SUCCESS;
            let context = unsafe {
                (opencl.cl_create_context)(null(), 1, &device, null_mut(), null_mut(), &mut err)
            };
            check_create("clCreateContext", err, context)?;
            let queue = unsafe { (opencl.cl_create_command_queue)(context, device, 0, &mut err) };
            check_create("clCreateCommandQueue", err, queue)?;
            let program = build_program(&opencl, context, device)?;
            let kernels = Kernels {
                embedding_weight: create_kernel(&opencl, program, "ds4_embedding_weight")?,
                rmsnorm_weight: create_kernel(&opencl, program, "ds4_rmsnorm_weight")?,
                matvec_weight: create_kernel(&opencl, program, "ds4_matvec_weight")?,
                embedding_f32: create_kernel(&opencl, program, "ds4_embedding_f32")?,
                rmsnorm_f32: create_kernel(&opencl, program, "ds4_rmsnorm_f32")?,
                matvec_f32: create_kernel(&opencl, program, "ds4_matvec_f32")?,
                add_inplace_f32: create_kernel(&opencl, program, "ds4_add_inplace_f32")?,
                add_scaled_inplace_f32: create_kernel(
                    &opencl,
                    program,
                    "ds4_add_scaled_inplace_f32",
                )?,
                silu_product_f32: create_kernel(&opencl, program, "ds4_silu_product_f32")?,
                rope_f32: create_kernel(&opencl, program, "ds4_rope_f32")?,
                store_cache_f32: create_kernel(&opencl, program, "ds4_store_cache_f32")?,
                attention_decode_f32: create_kernel(&opencl, program, "ds4_attention_decode_f32")?,
            };
            Ok(Arc::new(Self {
                opencl,
                context,
                queue,
                program,
                kernels,
                platform_name,
                device_name,
            }))
        }

        pub fn platform_name(&self) -> &str {
            &self.platform_name
        }

        pub fn device_name(&self) -> &str {
            &self.device_name
        }

        pub fn alloc(runtime: &Arc<Self>, bytes: usize) -> Ds4Result<ArcMem> {
            let len = bytes.max(1);
            let mut err = CL_SUCCESS;
            let mem = unsafe {
                (runtime.opencl.cl_create_buffer)(
                    runtime.context,
                    CL_MEM_READ_WRITE,
                    len,
                    null_mut(),
                    &mut err,
                )
            };
            check_create("clCreateBuffer", err, mem)?;
            Ok(ArcMem {
                runtime: Arc::clone(runtime),
                mem,
                bytes: len,
            })
        }

        pub fn from_bytes(runtime: &Arc<Self>, bytes: &[u8]) -> Ds4Result<ArcMem> {
            let len = bytes.len().max(1);
            let mut err = CL_SUCCESS;
            let host_ptr = if bytes.is_empty() {
                null_mut()
            } else {
                bytes.as_ptr().cast::<c_void>().cast_mut()
            };
            let flags = if bytes.is_empty() {
                CL_MEM_READ_WRITE
            } else {
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
            };
            let mem = unsafe {
                (runtime.opencl.cl_create_buffer)(runtime.context, flags, len, host_ptr, &mut err)
            };
            check_create("clCreateBuffer", err, mem)?;
            Ok(ArcMem {
                runtime: Arc::clone(runtime),
                mem,
                bytes: len,
            })
        }

        pub(crate) fn kernel(&self, id: KernelId) -> ClKernel {
            match id {
                KernelId::EmbeddingWeight => self.kernels.embedding_weight,
                KernelId::RmsNormWeight => self.kernels.rmsnorm_weight,
                KernelId::MatvecWeight => self.kernels.matvec_weight,
                KernelId::EmbeddingF32 => self.kernels.embedding_f32,
                KernelId::RmsNormF32 => self.kernels.rmsnorm_f32,
                KernelId::MatvecF32 => self.kernels.matvec_f32,
                KernelId::AddInplaceF32 => self.kernels.add_inplace_f32,
                KernelId::AddScaledInplaceF32 => self.kernels.add_scaled_inplace_f32,
                KernelId::SiluProductF32 => self.kernels.silu_product_f32,
                KernelId::RopeF32 => self.kernels.rope_f32,
                KernelId::StoreCacheF32 => self.kernels.store_cache_f32,
                KernelId::AttentionDecodeF32 => self.kernels.attention_decode_f32,
            }
        }

        pub(crate) fn set_arg_mem(
            &self,
            kernel: ClKernel,
            idx: u32,
            mem: &ArcMem,
        ) -> Ds4Result<()> {
            let raw = mem.mem;
            self.set_arg_raw(kernel, idx, &raw)
        }

        pub(crate) fn set_arg_u32(&self, kernel: ClKernel, idx: u32, value: u32) -> Ds4Result<()> {
            self.set_arg_raw(kernel, idx, &value)
        }

        pub(crate) fn set_arg_i32(&self, kernel: ClKernel, idx: u32, value: i32) -> Ds4Result<()> {
            self.set_arg_raw(kernel, idx, &value)
        }

        pub(crate) fn set_arg_f32(&self, kernel: ClKernel, idx: u32, value: f32) -> Ds4Result<()> {
            self.set_arg_raw(kernel, idx, &value)
        }

        pub(crate) fn launch_1d(&self, kernel: ClKernel, global: usize) -> Ds4Result<()> {
            let global = global.max(1);
            let local = if global >= 64 { 64usize } else { 1usize };
            let local_ptr = if global.is_multiple_of(local) {
                &local as *const usize
            } else {
                null()
            };
            let err = unsafe {
                (self.opencl.cl_enqueue_nd_range_kernel)(
                    self.queue,
                    kernel,
                    1,
                    null(),
                    &global,
                    local_ptr,
                    0,
                    null(),
                    null_mut(),
                )
            };
            check("clEnqueueNDRangeKernel", err)
        }

        pub fn finish(&self) -> Ds4Result<()> {
            let err = unsafe { (self.opencl.cl_finish)(self.queue) };
            check("clFinish", err)
        }

        fn set_arg_raw<T>(&self, kernel: ClKernel, idx: u32, value: &T) -> Ds4Result<()> {
            let err = unsafe {
                (self.opencl.cl_set_kernel_arg)(
                    kernel,
                    idx,
                    size_of::<T>(),
                    (value as *const T).cast::<c_void>(),
                )
            };
            check("clSetKernelArg", err)
        }
    }

    impl Drop for ArcRuntime {
        fn drop(&mut self) {
            unsafe {
                (self.opencl.cl_release_kernel)(self.kernels.embedding_weight);
                (self.opencl.cl_release_kernel)(self.kernels.rmsnorm_weight);
                (self.opencl.cl_release_kernel)(self.kernels.matvec_weight);
                (self.opencl.cl_release_kernel)(self.kernels.embedding_f32);
                (self.opencl.cl_release_kernel)(self.kernels.rmsnorm_f32);
                (self.opencl.cl_release_kernel)(self.kernels.matvec_f32);
                (self.opencl.cl_release_kernel)(self.kernels.add_inplace_f32);
                (self.opencl.cl_release_kernel)(self.kernels.add_scaled_inplace_f32);
                (self.opencl.cl_release_kernel)(self.kernels.silu_product_f32);
                (self.opencl.cl_release_kernel)(self.kernels.rope_f32);
                (self.opencl.cl_release_kernel)(self.kernels.store_cache_f32);
                (self.opencl.cl_release_kernel)(self.kernels.attention_decode_f32);
                (self.opencl.cl_release_program)(self.program);
                (self.opencl.cl_release_command_queue)(self.queue);
                (self.opencl.cl_release_context)(self.context);
            }
        }
    }

    pub struct ArcMem {
        runtime: Arc<ArcRuntime>,
        mem: ClMem,
        bytes: usize,
    }

    unsafe impl Send for ArcMem {}
    unsafe impl Sync for ArcMem {}

    impl ArcMem {
        pub fn bytes(&self) -> usize {
            self.bytes
        }

        pub fn write(&self, bytes: &[u8]) -> Ds4Result<()> {
            if bytes.len() > self.bytes {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::InvalidArgument,
                    format!(
                        "Arc buffer write too large: {} > {}",
                        bytes.len(),
                        self.bytes
                    ),
                ));
            }
            let err = unsafe {
                (self.runtime.opencl.cl_enqueue_write_buffer)(
                    self.runtime.queue,
                    self.mem,
                    CL_TRUE,
                    0,
                    bytes.len(),
                    bytes.as_ptr().cast::<c_void>(),
                    0,
                    null(),
                    null_mut(),
                )
            };
            check("clEnqueueWriteBuffer", err)
        }

        pub fn read(&self, bytes: &mut [u8]) -> Ds4Result<()> {
            if bytes.len() > self.bytes {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::InvalidArgument,
                    format!(
                        "Arc buffer read too large: {} > {}",
                        bytes.len(),
                        self.bytes
                    ),
                ));
            }
            let err = unsafe {
                (self.runtime.opencl.cl_enqueue_read_buffer)(
                    self.runtime.queue,
                    self.mem,
                    CL_TRUE,
                    0,
                    bytes.len(),
                    bytes.as_mut_ptr().cast::<c_void>(),
                    0,
                    null(),
                    null_mut(),
                )
            };
            check("clEnqueueReadBuffer", err)
        }
    }

    impl Drop for ArcMem {
        fn drop(&mut self) {
            unsafe {
                (self.runtime.opencl.cl_release_mem_object)(self.mem);
            }
        }
    }

    fn find_arc_device(opencl: &OpenCl) -> Ds4Result<(ClPlatformId, ClDeviceId, String, String)> {
        let mut platform_count = 0;
        check("clGetPlatformIDs", unsafe {
            (opencl.cl_get_platform_ids)(0, null_mut(), &mut platform_count)
        })?;
        if platform_count == 0 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "OpenCL reports no platforms",
            ));
        }
        let mut platforms = vec![null_mut(); platform_count as usize];
        check("clGetPlatformIDs", unsafe {
            (opencl.cl_get_platform_ids)(platform_count, platforms.as_mut_ptr(), null_mut())
        })?;
        for platform in platforms {
            let platform_name = platform_info(opencl, platform, CL_PLATFORM_NAME)?;
            let platform_vendor = platform_info(opencl, platform, CL_PLATFORM_VENDOR)?;
            if !platform_vendor.to_ascii_lowercase().contains("intel") {
                continue;
            }
            let mut device_count = 0;
            let err = unsafe {
                (opencl.cl_get_device_ids)(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    0,
                    null_mut(),
                    &mut device_count,
                )
            };
            if err != CL_SUCCESS || device_count == 0 {
                continue;
            }
            let mut devices = vec![null_mut(); device_count as usize];
            check("clGetDeviceIDs", unsafe {
                (opencl.cl_get_device_ids)(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    device_count,
                    devices.as_mut_ptr(),
                    null_mut(),
                )
            })?;
            for device in devices {
                let device_name = device_info(opencl, device, CL_DEVICE_NAME)?;
                let device_vendor = device_info(opencl, device, CL_DEVICE_VENDOR)?;
                let is_arc = device_name.to_ascii_lowercase().contains("arc");
                let is_intel = device_vendor.to_ascii_lowercase().contains("intel");
                if is_arc && is_intel {
                    return Ok((platform, device, platform_name, device_name));
                }
            }
        }
        Err(Ds4Error::new(
            Ds4ErrorKind::NotImplemented,
            "Intel Arc OpenCL GPU was not found",
        ))
    }

    fn platform_info(opencl: &OpenCl, platform: ClPlatformId, param: ClUint) -> Ds4Result<String> {
        let mut len = 0usize;
        check("clGetPlatformInfo", unsafe {
            (opencl.cl_get_platform_info)(platform, param, 0, null_mut(), &mut len)
        })?;
        let mut bytes = vec![0u8; len];
        check("clGetPlatformInfo", unsafe {
            (opencl.cl_get_platform_info)(
                platform,
                param,
                bytes.len(),
                bytes.as_mut_ptr().cast::<c_void>(),
                null_mut(),
            )
        })?;
        Ok(bytes_to_string(&bytes))
    }

    fn device_info(opencl: &OpenCl, device: ClDeviceId, param: ClUint) -> Ds4Result<String> {
        let mut len = 0usize;
        check("clGetDeviceInfo", unsafe {
            (opencl.cl_get_device_info)(device, param, 0, null_mut(), &mut len)
        })?;
        let mut bytes = vec![0u8; len];
        check("clGetDeviceInfo", unsafe {
            (opencl.cl_get_device_info)(
                device,
                param,
                bytes.len(),
                bytes.as_mut_ptr().cast::<c_void>(),
                null_mut(),
            )
        })?;
        Ok(bytes_to_string(&bytes))
    }

    fn build_program(
        opencl: &OpenCl,
        context: ClContext,
        device: ClDeviceId,
    ) -> Ds4Result<ClProgram> {
        let source = CString::new(ARC_KERNEL_SRC).expect("OpenCL source has no interior NUL");
        let ptr = source.as_ptr();
        let len = ARC_KERNEL_SRC.len();
        let mut err = CL_SUCCESS;
        let program =
            unsafe { (opencl.cl_create_program_with_source)(context, 1, &ptr, &len, &mut err) };
        check_create("clCreateProgramWithSource", err, program)?;
        let build_err = unsafe {
            (opencl.cl_build_program)(program, 1, &device, null(), null_mut(), null_mut())
        };
        if build_err != CL_SUCCESS {
            let log = build_log(opencl, program, device).unwrap_or_else(|_| String::new());
            unsafe {
                (opencl.cl_release_program)(program);
            }
            return Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                format!("Arc OpenCL program build failed ({build_err}): {log}"),
            ));
        }
        Ok(program)
    }

    fn create_kernel(opencl: &OpenCl, program: ClProgram, name: &str) -> Ds4Result<ClKernel> {
        let c_name = CString::new(name).expect("kernel name has no interior NUL");
        let mut err = CL_SUCCESS;
        let kernel = unsafe { (opencl.cl_create_kernel)(program, c_name.as_ptr(), &mut err) };
        check_create(name, err, kernel)
    }

    fn build_log(opencl: &OpenCl, program: ClProgram, device: ClDeviceId) -> Ds4Result<String> {
        let mut len = 0usize;
        check("clGetProgramBuildInfo", unsafe {
            (opencl.cl_get_program_build_info)(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                null_mut(),
                &mut len,
            )
        })?;
        let mut bytes = vec![0u8; len];
        check("clGetProgramBuildInfo", unsafe {
            (opencl.cl_get_program_build_info)(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                bytes.len(),
                bytes.as_mut_ptr().cast::<c_void>(),
                null_mut(),
            )
        })?;
        Ok(bytes_to_string(&bytes))
    }

    fn bytes_to_string(bytes: &[u8]) -> String {
        let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..end]).to_string()
    }

    fn check(name: &str, err: ClInt) -> Ds4Result<()> {
        if err == CL_SUCCESS {
            Ok(())
        } else {
            Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                format!("{name} failed with OpenCL error {err}"),
            ))
        }
    }

    fn check_create<T>(name: &str, err: ClInt, value: *mut T) -> Ds4Result<*mut T> {
        if err == CL_SUCCESS && !value.is_null() {
            Ok(value)
        } else {
            Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                format!("{name} failed with OpenCL error {err}"),
            ))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn arc_runtime_loads_or_reports_absent_arc() {
            match ArcRuntime::load() {
                Ok(runtime) => {
                    assert!(runtime.device_name().contains("Arc"));
                    assert!(runtime.platform_name().contains("OpenCL"));
                }
                Err(err) => assert_eq!(err.kind, Ds4ErrorKind::NotImplemented, "{err}"),
            }
        }
    }
}

#[cfg(windows)]
pub(crate) use imp::ClKernel;
pub use imp::{ArcMem, ArcRuntime, KernelId};

#[cfg(not(windows))]
mod fallback {
    use std::sync::Arc;

    use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

    #[derive(Debug, Clone, Copy)]
    pub enum KernelId {
        EmbeddingWeight,
        RmsNormWeight,
        MatvecWeight,
        EmbeddingF32,
        RmsNormF32,
        MatvecF32,
        AddInplaceF32,
        AddScaledInplaceF32,
        SiluProductF32,
        RopeF32,
        StoreCacheF32,
        AttentionDecodeF32,
    }

    pub(crate) type ClKernel = *mut std::ffi::c_void;

    pub struct ArcRuntime;
    pub struct ArcMem;

    impl ArcRuntime {
        pub fn load() -> Ds4Result<Arc<Self>> {
            Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "Intel Arc OpenCL backend currently loads on Windows",
            ))
        }
    }
}

#[cfg(not(windows))]
pub(crate) use fallback::ClKernel;
#[cfg(not(windows))]
pub use fallback::{ArcMem, ArcRuntime, KernelId};
