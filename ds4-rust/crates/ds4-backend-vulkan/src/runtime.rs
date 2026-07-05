#[cfg(windows)]
mod imp {
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::ptr::{null, null_mut};
    use std::sync::Arc;

    use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

    type VkResult = i32;
    type VkFlags = u32;
    type VkDeviceSize = u64;
    type VkInstance = *mut c_void;
    type VkPhysicalDevice = *mut c_void;
    type VkDevice = *mut c_void;
    type VkQueue = *mut c_void;
    type VkBuffer = u64;
    type VkDeviceMemory = u64;

    const VK_SUCCESS: VkResult = 0;
    const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
    const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
    const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
    const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
    const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
    const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
    const VK_QUEUE_COMPUTE_BIT: VkFlags = 0x0000_0002;
    const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkFlags = 0x0000_0001;
    const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkFlags = 0x0000_0002;
    const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkFlags = 0x0000_0020;
    const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
    const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkFlags = 0x0000_0001;
    const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkFlags = 0x0000_0002;
    const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkFlags = 0x0000_0004;
    const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
    const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;
    const VK_API_VERSION_1_0: u32 = 1 << 22;

    type VkCreateInstance = unsafe extern "system" fn(
        *const VkInstanceCreateInfo,
        *const c_void,
        *mut VkInstance,
    ) -> VkResult;
    type VkDestroyInstance = unsafe extern "system" fn(VkInstance, *const c_void);
    type VkEnumeratePhysicalDevices =
        unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult;
    type VkGetPhysicalDeviceProperties =
        unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties);
    type VkGetPhysicalDeviceMemoryProperties =
        unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties);
    type VkGetPhysicalDeviceQueueFamilyProperties =
        unsafe extern "system" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties);
    type VkCreateDevice = unsafe extern "system" fn(
        VkPhysicalDevice,
        *const VkDeviceCreateInfo,
        *const c_void,
        *mut VkDevice,
    ) -> VkResult;
    type VkDestroyDevice = unsafe extern "system" fn(VkDevice, *const c_void);
    type VkGetDeviceQueue = unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue);
    type VkCreateBuffer = unsafe extern "system" fn(
        VkDevice,
        *const VkBufferCreateInfo,
        *const c_void,
        *mut VkBuffer,
    ) -> VkResult;
    type VkDestroyBuffer = unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void);
    type VkGetBufferMemoryRequirements =
        unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements);
    type VkAllocateMemory = unsafe extern "system" fn(
        VkDevice,
        *const VkMemoryAllocateInfo,
        *const c_void,
        *mut VkDeviceMemory,
    ) -> VkResult;
    type VkFreeMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void);
    type VkBindBufferMemory =
        unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) -> VkResult;
    type VkMapMemory = unsafe extern "system" fn(
        VkDevice,
        VkDeviceMemory,
        VkDeviceSize,
        VkDeviceSize,
        VkFlags,
        *mut *mut c_void,
    ) -> VkResult;
    type VkUnmapMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory);

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    #[repr(C)]
    struct VkApplicationInfo {
        s_type: u32,
        p_next: *const c_void,
        p_application_name: *const c_char,
        application_version: u32,
        p_engine_name: *const c_char,
        engine_version: u32,
        api_version: u32,
    }

    #[repr(C)]
    struct VkInstanceCreateInfo {
        s_type: u32,
        p_next: *const c_void,
        flags: VkFlags,
        p_application_info: *const VkApplicationInfo,
        enabled_layer_count: u32,
        pp_enabled_layer_names: *const *const c_char,
        enabled_extension_count: u32,
        pp_enabled_extension_names: *const *const c_char,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct VkPhysicalDeviceProperties {
        api_version: u32,
        driver_version: u32,
        vendor_id: u32,
        device_id: u32,
        device_type: u32,
        device_name: [c_char; 256],
        pipeline_cache_uuid: [u8; 16],
        limits_and_sparse: [u64; 256],
    }

    impl Default for VkPhysicalDeviceProperties {
        fn default() -> Self {
            Self {
                api_version: 0,
                driver_version: 0,
                vendor_id: 0,
                device_id: 0,
                device_type: 0,
                device_name: [0; 256],
                pipeline_cache_uuid: [0; 16],
                limits_and_sparse: [0; 256],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct VkMemoryType {
        property_flags: VkFlags,
        heap_index: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct VkMemoryHeap {
        size: VkDeviceSize,
        flags: VkFlags,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct VkPhysicalDeviceMemoryProperties {
        memory_type_count: u32,
        memory_types: [VkMemoryType; 32],
        memory_heap_count: u32,
        memory_heaps: [VkMemoryHeap; 16],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct VkExtent3D {
        width: u32,
        height: u32,
        depth: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct VkQueueFamilyProperties {
        queue_flags: VkFlags,
        queue_count: u32,
        timestamp_valid_bits: u32,
        min_image_transfer_granularity: VkExtent3D,
    }

    #[repr(C)]
    struct VkDeviceQueueCreateInfo {
        s_type: u32,
        p_next: *const c_void,
        flags: VkFlags,
        queue_family_index: u32,
        queue_count: u32,
        p_queue_priorities: *const f32,
    }

    #[repr(C)]
    struct VkDeviceCreateInfo {
        s_type: u32,
        p_next: *const c_void,
        flags: VkFlags,
        queue_create_info_count: u32,
        p_queue_create_infos: *const VkDeviceQueueCreateInfo,
        enabled_layer_count: u32,
        pp_enabled_layer_names: *const *const c_char,
        enabled_extension_count: u32,
        pp_enabled_extension_names: *const *const c_char,
        p_enabled_features: *const c_void,
    }

    #[repr(C)]
    struct VkBufferCreateInfo {
        s_type: u32,
        p_next: *const c_void,
        flags: VkFlags,
        size: VkDeviceSize,
        usage: VkFlags,
        sharing_mode: u32,
        queue_family_index_count: u32,
        p_queue_family_indices: *const u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct VkMemoryRequirements {
        size: VkDeviceSize,
        alignment: VkDeviceSize,
        memory_type_bits: u32,
    }

    #[repr(C)]
    struct VkMemoryAllocateInfo {
        s_type: u32,
        p_next: *const c_void,
        allocation_size: VkDeviceSize,
        memory_type_index: u32,
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
            let c_name = CString::new(name).expect("Vulkan symbol has no interior NUL");
            let ptr = unsafe { GetProcAddress(self.handle, c_name.as_ptr()) };
            if ptr.is_null() {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::NotImplemented,
                    format!("Vulkan symbol {name} is unavailable"),
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

    struct Vulkan {
        _library: DynamicLibrary,
        vk_create_instance: VkCreateInstance,
        vk_destroy_instance: VkDestroyInstance,
        vk_enumerate_physical_devices: VkEnumeratePhysicalDevices,
        vk_get_physical_device_properties: VkGetPhysicalDeviceProperties,
        vk_get_physical_device_memory_properties: VkGetPhysicalDeviceMemoryProperties,
        vk_get_physical_device_queue_family_properties: VkGetPhysicalDeviceQueueFamilyProperties,
        vk_create_device: VkCreateDevice,
        vk_destroy_device: VkDestroyDevice,
        vk_get_device_queue: VkGetDeviceQueue,
        vk_create_buffer: VkCreateBuffer,
        vk_destroy_buffer: VkDestroyBuffer,
        vk_get_buffer_memory_requirements: VkGetBufferMemoryRequirements,
        vk_allocate_memory: VkAllocateMemory,
        vk_free_memory: VkFreeMemory,
        vk_bind_buffer_memory: VkBindBufferMemory,
        vk_map_memory: VkMapMemory,
        vk_unmap_memory: VkUnmapMemory,
    }

    impl Vulkan {
        fn load() -> Ds4Result<Self> {
            let library = DynamicLibrary::open("vulkan-1.dll")?;
            Ok(Self {
                vk_create_instance: library.get("vkCreateInstance")?,
                vk_destroy_instance: library.get("vkDestroyInstance")?,
                vk_enumerate_physical_devices: library.get("vkEnumeratePhysicalDevices")?,
                vk_get_physical_device_properties: library.get("vkGetPhysicalDeviceProperties")?,
                vk_get_physical_device_memory_properties: library
                    .get("vkGetPhysicalDeviceMemoryProperties")?,
                vk_get_physical_device_queue_family_properties: library
                    .get("vkGetPhysicalDeviceQueueFamilyProperties")?,
                vk_create_device: library.get("vkCreateDevice")?,
                vk_destroy_device: library.get("vkDestroyDevice")?,
                vk_get_device_queue: library.get("vkGetDeviceQueue")?,
                vk_create_buffer: library.get("vkCreateBuffer")?,
                vk_destroy_buffer: library.get("vkDestroyBuffer")?,
                vk_get_buffer_memory_requirements: library.get("vkGetBufferMemoryRequirements")?,
                vk_allocate_memory: library.get("vkAllocateMemory")?,
                vk_free_memory: library.get("vkFreeMemory")?,
                vk_bind_buffer_memory: library.get("vkBindBufferMemory")?,
                vk_map_memory: library.get("vkMapMemory")?,
                vk_unmap_memory: library.get("vkUnmapMemory")?,
                _library: library,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct VulkanMemoryHeap {
        pub size_bytes: u64,
        pub flags: u32,
    }

    #[derive(Debug, Clone)]
    pub struct VulkanDeviceInfo {
        pub device_name: String,
        pub vendor_id: u32,
        pub device_id: u32,
        pub device_type: String,
        pub api_version: u32,
        pub driver_version: u32,
        pub compute_queue_family: u32,
        pub memory_heaps: Vec<VulkanMemoryHeap>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VulkanMemoryKind {
        HostVisible,
        DeviceLocal,
    }

    pub struct VulkanRuntime {
        vulkan: Vulkan,
        instance: VkInstance,
        physical_device: VkPhysicalDevice,
        device: VkDevice,
        queue: VkQueue,
        queue_family_index: u32,
        memory_properties: VkPhysicalDeviceMemoryProperties,
        device_info: VulkanDeviceInfo,
    }

    unsafe impl Send for VulkanRuntime {}
    unsafe impl Sync for VulkanRuntime {}

    impl VulkanRuntime {
        pub fn load() -> Ds4Result<Arc<Self>> {
            let vulkan = Vulkan::load()?;
            let instance = create_instance(&vulkan)?;
            let selected = match select_physical_device(&vulkan, instance) {
                Ok(selected) => selected,
                Err(err) => {
                    unsafe {
                        (vulkan.vk_destroy_instance)(instance, null());
                    }
                    return Err(err);
                }
            };
            let (device, queue) =
                match create_device(&vulkan, selected.physical_device, selected.queue_family) {
                    Ok(pair) => pair,
                    Err(err) => {
                        unsafe {
                            (vulkan.vk_destroy_instance)(instance, null());
                        }
                        return Err(err);
                    }
                };
            Ok(Arc::new(Self {
                vulkan,
                instance,
                physical_device: selected.physical_device,
                device,
                queue,
                queue_family_index: selected.queue_family,
                memory_properties: selected.memory_properties,
                device_info: selected.info,
            }))
        }

        pub fn device_info(&self) -> &VulkanDeviceInfo {
            &self.device_info
        }

        pub fn device_name(&self) -> &str {
            &self.device_info.device_name
        }

        pub fn queue_family_index(&self) -> u32 {
            self.queue_family_index
        }

        pub fn alloc_host_visible(runtime: &Arc<Self>, bytes: usize) -> Ds4Result<VulkanBuffer> {
            runtime.alloc_buffer(
                bytes,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VulkanMemoryKind::HostVisible,
            )
        }

        pub fn alloc_device_local(runtime: &Arc<Self>, bytes: usize) -> Ds4Result<VulkanBuffer> {
            runtime.alloc_buffer(
                bytes,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VulkanMemoryKind::DeviceLocal,
            )
        }

        fn alloc_buffer(
            self: &Arc<Self>,
            bytes: usize,
            required_flags: VkFlags,
            memory_kind: VulkanMemoryKind,
        ) -> Ds4Result<VulkanBuffer> {
            let len = bytes.max(1) as VkDeviceSize;
            let create_info = VkBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: null(),
                flags: 0,
                size: len,
                usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: null(),
            };
            let mut buffer = 0;
            check("vkCreateBuffer", unsafe {
                (self.vulkan.vk_create_buffer)(self.device, &create_info, null(), &mut buffer)
            })?;
            let mut requirements = VkMemoryRequirements::default();
            unsafe {
                (self.vulkan.vk_get_buffer_memory_requirements)(
                    self.device,
                    buffer,
                    &mut requirements,
                );
            }
            let memory_type_index =
                match self.find_memory_type(requirements.memory_type_bits, required_flags) {
                    Ok(index) => index,
                    Err(err) => {
                        unsafe {
                            (self.vulkan.vk_destroy_buffer)(self.device, buffer, null());
                        }
                        return Err(err);
                    }
                };
            let alloc_info = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: null(),
                allocation_size: requirements.size,
                memory_type_index,
            };
            let mut memory = 0;
            if let Err(err) = check("vkAllocateMemory", unsafe {
                (self.vulkan.vk_allocate_memory)(self.device, &alloc_info, null(), &mut memory)
            }) {
                unsafe {
                    (self.vulkan.vk_destroy_buffer)(self.device, buffer, null());
                }
                return Err(err);
            }
            if let Err(err) = check("vkBindBufferMemory", unsafe {
                (self.vulkan.vk_bind_buffer_memory)(self.device, buffer, memory, 0)
            }) {
                unsafe {
                    (self.vulkan.vk_free_memory)(self.device, memory, null());
                    (self.vulkan.vk_destroy_buffer)(self.device, buffer, null());
                }
                return Err(err);
            }
            Ok(VulkanBuffer {
                runtime: Arc::clone(self),
                buffer,
                memory,
                bytes: len as usize,
                memory_kind,
            })
        }

        fn find_memory_type(
            &self,
            memory_type_bits: u32,
            required_flags: VkFlags,
        ) -> Ds4Result<u32> {
            for idx in 0..self.memory_properties.memory_type_count.min(32) {
                let supported = (memory_type_bits & (1 << idx)) != 0;
                let flags = self.memory_properties.memory_types[idx as usize].property_flags;
                if supported && (flags & required_flags) == required_flags {
                    return Ok(idx);
                }
            }
            Err(Ds4Error::new(
                Ds4ErrorKind::OutOfMemory,
                format!(
                    "Vulkan memory type not found for flags 0x{required_flags:08x} and mask 0x{memory_type_bits:08x}"
                ),
            ))
        }
    }

    impl Drop for VulkanRuntime {
        fn drop(&mut self) {
            unsafe {
                let _ = self.physical_device;
                let _ = self.queue;
                (self.vulkan.vk_destroy_device)(self.device, null());
                (self.vulkan.vk_destroy_instance)(self.instance, null());
            }
        }
    }

    pub struct VulkanBuffer {
        runtime: Arc<VulkanRuntime>,
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        bytes: usize,
        memory_kind: VulkanMemoryKind,
    }

    unsafe impl Send for VulkanBuffer {}
    unsafe impl Sync for VulkanBuffer {}

    impl VulkanBuffer {
        pub fn bytes(&self) -> usize {
            self.bytes
        }

        pub fn memory_kind(&self) -> VulkanMemoryKind {
            self.memory_kind
        }

        pub fn write(&self, bytes: &[u8]) -> Ds4Result<()> {
            if bytes.len() > self.bytes {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::InvalidArgument,
                    format!(
                        "Vulkan buffer write too large: {} > {}",
                        bytes.len(),
                        self.bytes
                    ),
                ));
            }
            if self.memory_kind != VulkanMemoryKind::HostVisible {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::Backend,
                    "Vulkan buffer is device-local and cannot be mapped by the host",
                ));
            }
            let mapped = self.map(bytes.len())?;
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
                (self.runtime.vulkan.vk_unmap_memory)(self.runtime.device, self.memory);
            }
            Ok(())
        }

        pub fn read(&self, bytes: &mut [u8]) -> Ds4Result<()> {
            if bytes.len() > self.bytes {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::InvalidArgument,
                    format!(
                        "Vulkan buffer read too large: {} > {}",
                        bytes.len(),
                        self.bytes
                    ),
                ));
            }
            if self.memory_kind != VulkanMemoryKind::HostVisible {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::Backend,
                    "Vulkan buffer is device-local and cannot be mapped by the host",
                ));
            }
            let mapped = self.map(bytes.len())?;
            unsafe {
                std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), bytes.as_mut_ptr(), bytes.len());
                (self.runtime.vulkan.vk_unmap_memory)(self.runtime.device, self.memory);
            }
            Ok(())
        }

        fn map(&self, len: usize) -> Ds4Result<*mut c_void> {
            let mut mapped = null_mut();
            check("vkMapMemory", unsafe {
                (self.runtime.vulkan.vk_map_memory)(
                    self.runtime.device,
                    self.memory,
                    0,
                    len as VkDeviceSize,
                    0,
                    &mut mapped,
                )
            })?;
            if mapped.is_null() {
                return Err(Ds4Error::new(
                    Ds4ErrorKind::Backend,
                    "vkMapMemory returned a null host pointer",
                ));
            }
            Ok(mapped)
        }
    }

    impl Drop for VulkanBuffer {
        fn drop(&mut self) {
            unsafe {
                (self.runtime.vulkan.vk_destroy_buffer)(self.runtime.device, self.buffer, null());
                (self.runtime.vulkan.vk_free_memory)(self.runtime.device, self.memory, null());
            }
        }
    }

    struct SelectedDevice {
        physical_device: VkPhysicalDevice,
        queue_family: u32,
        memory_properties: VkPhysicalDeviceMemoryProperties,
        info: VulkanDeviceInfo,
    }

    fn create_instance(vulkan: &Vulkan) -> Ds4Result<VkInstance> {
        let app_name = CString::new("ds4-vulkan").expect("static name has no interior NUL");
        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: null(),
            p_application_name: app_name.as_ptr(),
            application_version: 1,
            p_engine_name: app_name.as_ptr(),
            engine_version: 1,
            api_version: VK_API_VERSION_1_0,
        };
        let create_info = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: null(),
        };
        let mut instance = null_mut();
        check("vkCreateInstance", unsafe {
            (vulkan.vk_create_instance)(&create_info, null(), &mut instance)
        })?;
        if instance.is_null() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                "vkCreateInstance returned a null instance",
            ));
        }
        Ok(instance)
    }

    fn select_physical_device(vulkan: &Vulkan, instance: VkInstance) -> Ds4Result<SelectedDevice> {
        let mut count = 0;
        check("vkEnumeratePhysicalDevices", unsafe {
            (vulkan.vk_enumerate_physical_devices)(instance, &mut count, null_mut())
        })?;
        if count == 0 {
            return Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "Vulkan reports no physical devices",
            ));
        }
        let mut devices = vec![null_mut(); count as usize];
        check("vkEnumeratePhysicalDevices", unsafe {
            (vulkan.vk_enumerate_physical_devices)(instance, &mut count, devices.as_mut_ptr())
        })?;
        let mut best: Option<(u32, usize, SelectedDevice)> = None;
        for (idx, physical_device) in devices.into_iter().enumerate() {
            if physical_device.is_null() {
                continue;
            }
            let mut properties = VkPhysicalDeviceProperties::default();
            unsafe {
                (vulkan.vk_get_physical_device_properties)(physical_device, &mut properties);
            }
            let Some(queue_family) = compute_queue_family(vulkan, physical_device)? else {
                continue;
            };
            let mut memory_properties = VkPhysicalDeviceMemoryProperties::default();
            unsafe {
                (vulkan.vk_get_physical_device_memory_properties)(
                    physical_device,
                    &mut memory_properties,
                );
            }
            let info = device_info(&properties, &memory_properties, queue_family);
            let score = device_score(&info, properties.device_type);
            let selected = SelectedDevice {
                physical_device,
                queue_family,
                memory_properties,
                info,
            };
            match best.as_ref() {
                Some((best_score, best_idx, _))
                    if (*best_score, usize::MAX - *best_idx) >= (score, usize::MAX - idx) => {}
                _ => best = Some((score, idx, selected)),
            }
        }
        best.map(|(_, _, selected)| selected).ok_or_else(|| {
            Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "Vulkan reports no compute-capable physical devices",
            )
        })
    }

    fn compute_queue_family(
        vulkan: &Vulkan,
        physical_device: VkPhysicalDevice,
    ) -> Ds4Result<Option<u32>> {
        let mut count = 0;
        unsafe {
            (vulkan.vk_get_physical_device_queue_family_properties)(
                physical_device,
                &mut count,
                null_mut(),
            );
        }
        if count == 0 {
            return Ok(None);
        }
        let mut families = vec![VkQueueFamilyProperties::default(); count as usize];
        unsafe {
            (vulkan.vk_get_physical_device_queue_family_properties)(
                physical_device,
                &mut count,
                families.as_mut_ptr(),
            );
        }
        Ok(families.iter().enumerate().find_map(|(idx, family)| {
            ((family.queue_count > 0) && (family.queue_flags & VK_QUEUE_COMPUTE_BIT != 0))
                .then_some(idx as u32)
        }))
    }

    fn create_device(
        vulkan: &Vulkan,
        physical_device: VkPhysicalDevice,
        queue_family_index: u32,
    ) -> Ds4Result<(VkDevice, VkQueue)> {
        let priority = 1.0f32;
        let queue_info = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: &priority,
        };
        let create_info = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: null(),
            p_enabled_features: null(),
        };
        let mut device = null_mut();
        check("vkCreateDevice", unsafe {
            (vulkan.vk_create_device)(physical_device, &create_info, null(), &mut device)
        })?;
        if device.is_null() {
            return Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                "vkCreateDevice returned a null device",
            ));
        }
        let mut queue = null_mut();
        unsafe {
            (vulkan.vk_get_device_queue)(device, queue_family_index, 0, &mut queue);
        }
        if queue.is_null() {
            unsafe {
                (vulkan.vk_destroy_device)(device, null());
            }
            return Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                "vkGetDeviceQueue returned a null compute queue",
            ));
        }
        Ok((device, queue))
    }

    fn device_info(
        properties: &VkPhysicalDeviceProperties,
        memory_properties: &VkPhysicalDeviceMemoryProperties,
        compute_queue_family: u32,
    ) -> VulkanDeviceInfo {
        let memory_heaps = memory_properties.memory_heaps
            [..memory_properties.memory_heap_count.min(16) as usize]
            .iter()
            .map(|heap| VulkanMemoryHeap {
                size_bytes: heap.size,
                flags: heap.flags,
            })
            .collect::<Vec<_>>();
        VulkanDeviceInfo {
            device_name: c_string_to_string(properties.device_name.as_ptr()),
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_type: device_type_name(properties.device_type).to_string(),
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            compute_queue_family,
            memory_heaps,
        }
    }

    fn device_score(info: &VulkanDeviceInfo, device_type: u32) -> u32 {
        let name = info.device_name.to_ascii_lowercase();
        if name.contains("arc") {
            300
        } else if device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU {
            200
        } else if device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU {
            100
        } else {
            0
        }
    }

    fn device_type_name(device_type: u32) -> &'static str {
        match device_type {
            VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => "integrated-gpu",
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => "discrete-gpu",
            3 => "virtual-gpu",
            4 => "cpu",
            _ => "other",
        }
    }

    fn c_string_to_string(ptr: *const c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    fn check(name: &str, result: VkResult) -> Ds4Result<()> {
        if result == VK_SUCCESS {
            Ok(())
        } else {
            Err(Ds4Error::new(
                Ds4ErrorKind::Backend,
                format!("{name} failed with Vulkan result {result}"),
            ))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn vulkan_runtime_loads_or_reports_absent_vulkan() {
            match VulkanRuntime::load() {
                Ok(runtime) => {
                    assert!(!runtime.device_name().is_empty());
                    assert!(!runtime.device_info().memory_heaps.is_empty());
                }
                Err(err) => assert!(
                    matches!(
                        err.kind,
                        Ds4ErrorKind::NotImplemented | Ds4ErrorKind::Backend
                    ),
                    "{err}"
                ),
            }
        }

        #[test]
        fn host_visible_buffer_roundtrip_or_reports_absent_vulkan() {
            match VulkanRuntime::load() {
                Ok(runtime) => {
                    let buffer = VulkanRuntime::alloc_host_visible(&runtime, 16).unwrap();
                    assert_eq!(buffer.memory_kind(), VulkanMemoryKind::HostVisible);
                    buffer.write(b"ds4-vulkan").unwrap();
                    let mut out = [0u8; 10];
                    buffer.read(&mut out).unwrap();
                    assert_eq!(&out, b"ds4-vulkan");
                }
                Err(err) => assert!(
                    matches!(
                        err.kind,
                        Ds4ErrorKind::NotImplemented | Ds4ErrorKind::Backend
                    ),
                    "{err}"
                ),
            }
        }

        #[test]
        fn device_local_buffer_allocates_or_reports_absent_vulkan() {
            match VulkanRuntime::load() {
                Ok(runtime) => {
                    let buffer = VulkanRuntime::alloc_device_local(&runtime, 4096).unwrap();
                    assert_eq!(buffer.memory_kind(), VulkanMemoryKind::DeviceLocal);
                    assert!(buffer.bytes() >= 4096);
                }
                Err(err) => assert!(
                    matches!(
                        err.kind,
                        Ds4ErrorKind::NotImplemented | Ds4ErrorKind::Backend
                    ),
                    "{err}"
                ),
            }
        }
    }
}

#[cfg(windows)]
pub use imp::{VulkanBuffer, VulkanDeviceInfo, VulkanMemoryHeap, VulkanMemoryKind, VulkanRuntime};

#[cfg(not(windows))]
mod fallback {
    use std::sync::Arc;

    use ds4_types::{Ds4Error, Ds4ErrorKind, Ds4Result};

    #[derive(Debug, Clone)]
    pub struct VulkanMemoryHeap {
        pub size_bytes: u64,
        pub flags: u32,
    }

    #[derive(Debug, Clone)]
    pub struct VulkanDeviceInfo {
        pub device_name: String,
        pub vendor_id: u32,
        pub device_id: u32,
        pub device_type: String,
        pub api_version: u32,
        pub driver_version: u32,
        pub compute_queue_family: u32,
        pub memory_heaps: Vec<VulkanMemoryHeap>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VulkanMemoryKind {
        HostVisible,
        DeviceLocal,
    }

    pub struct VulkanRuntime;
    pub struct VulkanBuffer;

    impl VulkanRuntime {
        pub fn load() -> Ds4Result<Arc<Self>> {
            Err(Ds4Error::new(
                Ds4ErrorKind::NotImplemented,
                "Vulkan backend currently loads on Windows",
            ))
        }
    }
}

#[cfg(not(windows))]
pub use fallback::{
    VulkanBuffer, VulkanDeviceInfo, VulkanMemoryHeap, VulkanMemoryKind, VulkanRuntime,
};
