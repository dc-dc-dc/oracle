#!/usr/bin/python3
import os, platform, json, argparse, ctypes, sys

OSX=platform.system() == "Darwin"
WINDOWS=platform.system() == "Windows"
LOG_ERROR=False

def extract_val(val): return ([x for x in val] if hasattr(val, "__len__") else val)
def getdict(struct): return dict((field, extract_val(getattr(struct, field))) for field, _ in struct._fields_)
def ascii_str(s): return ''.join(chr(i) for i in s)

def error_wrap(method: str, c: int):
    if c != 0:
        if LOG_ERROR: sys.stderr.write(f"Error: {method} failed with error code {c}\n")
        return None

def rocm():
    pass

def metal():
    pass

class cudaDeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char*256),
        ("uuid", ctypes.c_char*16),
        ("luid", ctypes.c_char*8),
        ("luidDeviceNodeMask", ctypes.c_uint),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int*3),
        ("maxGridSize", ctypes.c_int*3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("textureAlignment", ctypes.c_size_t),
        ("texturePitchAlignment", ctypes.c_size_t),
        ("deviceOverlap", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("kernelExecTimeoutEnabled", ctypes.c_int),
        ("integrated", ctypes.c_int),
        ("canMapHostMemory", ctypes.c_int),
        ("computeMode", ctypes.c_int),
        ("maxTexture1D", ctypes.c_int),
        ("maxTexture1DMipmap", ctypes.c_int),
        ("maxTexture1DLinear", ctypes.c_int),
        ("maxTexture2D", ctypes.c_int*2),
        ("maxTexture2DMipmap", ctypes.c_int*2),
        ("maxTexture2DLinear", ctypes.c_int*3),
        ("maxTexture2DGather", ctypes.c_int*2),
        ("maxTexture3D", ctypes.c_int*3),
        ("maxTexture3DAlt", ctypes.c_int*3),
        ("maxTextureCubemap", ctypes.c_int),
        ("maxTexture1DLayered", ctypes.c_int*2),
        ("maxTexture2DLayered", ctypes.c_int*3),
        ("maxTextureCubemapLayered", ctypes.c_int*2),
        ("maxSurface1D", ctypes.c_int),
        ("maxSurface2D", ctypes.c_int*2),
        ("maxSurface3D", ctypes.c_int*3),
        ("maxSurface1DLayered", ctypes.c_int*2),
        ("maxSurface2DLayered", ctypes.c_int*3),
        ("maxSurfaceCubemap", ctypes.c_int),
        ("maxSurfaceCubemapLayered", ctypes.c_int*2),
        ("surfaceAlignment", ctypes.c_size_t),
        ("concurrentKernels", ctypes.c_int),
        ("ECCEnabled", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("pciDomainID", ctypes.c_int),
        ("tccDriver", ctypes.c_int),
        ("asyncEngineCount", ctypes.c_int),
        ("unifiedAddressing", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
        ("persistingL2CacheMaxSize", ctypes.c_int),
        ("maxThreadsPerMultiProcessor", ctypes.c_int),
        ("streamPrioritiesSupported", ctypes.c_int),
        ("globalL1CacheSupported", ctypes.c_int),
        ("localL1CacheSupported", ctypes.c_int),
        ("sharedMemPerMultiprocessor", ctypes.c_size_t),
        ("regsPerMultiprocessor", ctypes.c_int),
        ("managedMemory", ctypes.c_int),
        ("isMultiGpuBoard", ctypes.c_int),
        ("multiGpuBoardGroupID", ctypes.c_int),
        ("hostNativeAtomicSupported", ctypes.c_int),
        ("singleToDoublePrecisionPerfRatio", ctypes.c_int),
        ("pageableMemoryAccess", ctypes.c_int),
        ("concurrentManagedAccess", ctypes.c_int),
        ("computePreemptionSupported", ctypes.c_int),
        ("canUseHostPointerForRegisteredMem", ctypes.c_int),
        ("cooperativeLaunch", ctypes.c_int),
        ("cooperativeMultiDeviceLaunch", ctypes.c_int),
        ("sharedMemPerBlockOptin", ctypes.c_size_t),
        ("pageableMemoryAccessUsesHostPageTables", ctypes.c_int),
        ("directManagedMemAccessFromHost", ctypes.c_int),
        ("maxBlocksPerMultiProcessor", ctypes.c_int),
        ("accessPolicyMaxWindowSize", ctypes.c_int),
        ("reservedSharedMemPerBlock", ctypes.c_size_t),
        ("hostRegisterSupported", ctypes.c_int),
        ("sparseCudaArraySupported", ctypes.c_int),
        ("hostRegisterReadOnlySupported", ctypes.c_int),
        ("timelineSemaphoreInteropSupported", ctypes.c_int),
        ("memoryPoolsSupported", ctypes.c_int),
        ("gpuDirectRDMASupported", ctypes.c_int),
        ("gpuDirectRDMAFlushWritesOptions", ctypes.c_uint),
        ("gpuDirectRDMAWritesOrdering", ctypes.c_int),
        ("memoryPoolSupportedHandleTypes", ctypes.c_uint),
        ("deferredMappingCudaArraySupported", ctypes.c_int),
        ("ipcEventSupported", ctypes.c_int),
        ("clusterLaunch", ctypes.c_int),
        ("unifiedFunctionPointers", ctypes.c_int),
        ("reserved", ctypes.c_int*63)
    ]

def cuda():
    try:
        cuda_path = os.getenv("CUDA_PATH", None)
        if not cuda_path: 
            if LOG_ERROR: sys.stderr.write("CUDA_PATH not set\n")
            return None
        cuda = ctypes.cdll.LoadLibrary(os.path.join(cuda_path, "bin", "cudart64_12.dll") if WINDOWS else "libcuda.so")
        _cudaGetDeviceCount = cuda["cudaGetDeviceCount"]
        _cudaGetDeviceProperties = cuda["cudaGetDeviceProperties"]
        device_count = ctypes.c_int()
        error_wrap("cudaGetDeviceCount", _cudaGetDeviceCount(ctypes.byref(device_count)))
        _devices = dict()
        for i in range(device_count.value):
            props = cudaDeviceProp()
            error_wrap("cudaGetDeviceProperties", _cudaGetDeviceProperties(ctypes.byref(props), i))
            temp = getdict(props)
            temp["name"] = ascii_str(temp["name"])
            temp['uuid'] = ascii_str(temp['uuid'])
            temp['luid'] = ascii_str(temp['luid'])
            temp["reserved"] = None
            _devices[i] = temp
        return _devices
    except Exception as e:
        if LOG_ERROR: sys.stderr.write(f"{e}\n")
        return None


class nvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]
class nvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]
class nvmlProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]
# gets nvidia gpu information
def nvml():
    try:
        _nvml = ctypes.cdll.LoadLibrary(os.path.join(os.getenv("WINDIR", "C:/Windows"), "System32/nvml.dll") if WINDOWS else "libnvidia-ml.so.1")
        _nvmlInit = _nvml["nvmlInit_v2"]
        _nvmlShutdown = _nvml["nvmlShutdown"]
        _nvmlDeviceGetCount = _nvml["nvmlDeviceGetCount_v2"]
        _nvmlDeviceGetHandleByIndex = _nvml["nvmlDeviceGetHandleByIndex_v2"]
        _nvmlDeviceGetUtilizationRates = _nvml["nvmlDeviceGetUtilizationRates"]
        _nvmlDeviceGetMemoryInfo = _nvml["nvmlDeviceGetMemoryInfo"]
        _nvmlDeviceGetPowerUsage = _nvml["nvmlDeviceGetPowerUsage"]
        _nvmlDeviceGetComputeRunningProcesses = _nvml["nvmlDeviceGetComputeRunningProcesses_v3"]
        _nvmlSystemGetProcessName = _nvml["nvmlSystemGetProcessName"]

        error_wrap("nvmlInit", _nvmlInit())
        device_count = ctypes.c_uint()
        error_wrap("nvmlDeviceGetCount", _nvmlDeviceGetCount(ctypes.byref(device_count)))
        res = dict()
        for i in range(device_count.value):
            handle = ctypes.c_void_p()
            error_wrap("nvmlDeviceGetHandleByIndex", _nvmlDeviceGetHandleByIndex(i, ctypes.byref(handle)))
            _utilization = nvmlUtilization()
            error_wrap("nvmlDeviceGetUtilizationRates", _nvmlDeviceGetUtilizationRates(handle, ctypes.byref(_utilization)))
            _memory = nvmlMemory()
            error_wrap("nvmlDeviceGetMemoryInfo", _nvmlDeviceGetMemoryInfo(handle, ctypes.byref(_memory)))
            _power = ctypes.c_uint()
            error_wrap("nvmlDeviceGetPowerUsage", _nvmlDeviceGetPowerUsage(handle, ctypes.byref(_power)))
            _infoCount = ctypes.c_uint();
            _infos = ctypes.pointer(nvmlProcessInfo());
            error_wrap("nvmlDeviceGetComputeRunningProcesses", _nvmlDeviceGetComputeRunningProcesses(handle, ctypes.byref(_infoCount), _infos));
            for i in range(_infoCount.value):
                if _infos[i].pid == 0: continue
                _name = (ctypes.c_char * 256)()
                error_wrap("nvmlSystemGetProcessName", _nvmlSystemGetProcessName(_infos[i].pid, _name, 256))
                print(_name.value)
                print(f"pid: {_infos[i].pid} usedGpuMemory: {_infos[i].usedGpuMemory}, name: {_name.value.decode('utf-8')}")
            res[i] = {
                "utilization": getdict(_utilization),
                "memory": getdict(_memory),
                "power": _power.value,
            }
        error_wrap("nvmlShutdown", _nvmlShutdown())
        return res
    except Exception as e:
        if LOG_ERROR: sys.stderr.write(f"{e}\n")
        return None

cl_device_map = {
    1: "CL_DEVICE_TYPE_DEFAULT",
    2: "CL_DEVICE_TYPE_CPU",
    4: "CL_DEVICE_TYPE_GPU",
    8: "CL_DEVICE_TYPE_ACCELERATOR",
    16: "CL_DEVICE_TYPE_CUSTOM",
}

cl_cache_map = {
    0: "CL_NONE",
    1: "CL_READ_ONLY_CACHE",
    2: "CL_READ_WRITE_CACHE",
}

cl_queue_map = {
    1: "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE",
    2: "CL_QUEUE_PROFILING_ENABLE",
}

def __clGetPlatformInfo(_clGetPlatformInfo, platform_id, param_name):
    name_size = ctypes.c_uint()
    error_wrap("clGetPlatformInfo", _clGetPlatformInfo(platform_id, param_name, None, None, ctypes.byref(name_size)))
    name = (ctypes.c_char * name_size.value)()
    error_wrap("clGetPlatformInfo", _clGetPlatformInfo(platform_id, param_name, name_size.value, ctypes.byref(name), None))
    return name.value.decode("utf-8")


def __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, size, ctype: ctypes._SimpleCData): 
    error_wrap("clGetDeviceInfo", _clGetDeviceInfo(device_id, param_name, size, ctypes.byref(ctype), None))
    return ctype

def __clGetDeviceInfoStr(_clGetDeviceInfo, device_id, param_name):
    name_size = ctypes.c_uint()
    error_wrap("clGetDeviceInfo", _clGetDeviceInfo(device_id, param_name, None, None, ctypes.byref(name_size)))
    return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, name_size.value, (ctypes.c_char * name_size.value)()).value.decode("utf-8").strip()

def __clGetDeviceInfoInt(_clGetDeviceInfo, device_id, param_name): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 8, ctypes.c_uint()).value
def __clGetDeviceInfoBool(_clGetDeviceInfo, device_id, param_name): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 4, ctypes.c_bool()).value
def __clGetDeviceInfoSize(_clGetDeviceInfo, device_id, param_name): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 16, ctypes.c_size_t()).value
def __clGetDeviceInfoLong(_clGetDeviceInfo, device_id, param_name): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 16, ctypes.c_long()).value
def __clGetDeviceInfoLongArr(_clGetDeviceInfo, device_id, param_name, size): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 16*size, (ctypes.c_long*size)())[:]


def opencl():
    try:
        opencl = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/OpenCL.framework/OpenCL" if OSX else "libOpenCL.so")
        _clGetDeviceInfo = opencl["clGetDeviceInfo"]
        _clGetDeviceIDs = opencl["clGetDeviceIDs"]
        _clGetPlatformIDs = opencl["clGetPlatformIDs"]
        _clGetPlatformInfo = opencl["clGetPlatformInfo"]
        platform_count = ctypes.c_uint()
        error_wrap("clGetPlatformIDs", _clGetPlatformIDs(0, None, ctypes.byref(platform_count))) 
        platform_ids = (ctypes.c_uint * platform_count.value)()
        error_wrap("clGetPlatformIDs", _clGetPlatformIDs(platform_count.value, ctypes.byref(platform_ids), None))
        _platforms = dict()
        for platform_id in platform_ids:
            temp = {
                "profile": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0900),
                "version": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0901),
                "vendor": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0903),
                "name": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0902),
                "extensions": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0904).split(" "),
            }
            device_count = ctypes.c_uint()
            error_wrap("clGetDeviceIDs", _clGetDeviceIDs(platform_id, 0xFFFFFFFF, 0, None, ctypes.byref(device_count)))
            devices = (ctypes.c_uint * device_count.value)()
            error_wrap("clGetDeviceIDs", _clGetDeviceIDs(platform_id, 0xFFFFFFFF, device_count.value, ctypes.byref(devices), None))
            _devices = dict()
            for i in devices:
                _devices[i] = {
                    "name": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102B),
                    "vendor": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102C),
                    "vendor_id": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1001),
                    "version": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102F),
                    "driver": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102D),
                    "profile": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102E),
                    "opencl_c_version": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x103D),
                    "ecc_enabled": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1024),
                    "extensions": [i for i in __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x1030).split(" ") if i],
                    "type": cl_device_map[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1000)],
                    "max_compute_units": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1002),
                    "max_work_item_dimensions": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1003),
                    "max_work_group_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1004),
                    "max_work_item_size": __clGetDeviceInfoLongArr(_clGetDeviceInfo, i, 0x1005, 3),
                    "max_clock_frequency": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100C),
                    "address_bits": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100D),
                    "device_available": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1027),
                    "compiler_available": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1028),
                    "linker_available": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x103A),
                    "endian_little": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1026),
                    "image_support": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1016),
                    "max_read_image_args": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100E),
                    "max_write_image_args": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100F),
                    "image2d_max_width": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1011),
                    "image2d_max_height": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1012),
                    "image3d_max_width": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1013),
                    "image3d_max_height": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1014),
                    "image3d_max_depth": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1015),
                    "image_max_buffer_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1040),
                    "image_max_array_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1041),
                    "image_pitch_alignment": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x104A),
                    "image_base_address_alignment": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x104B),
                    "host_unified_memory": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1035),
                    "preferred_interop_user_sync": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1048),
                    "mem_base_addr_align": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1019),
                    "min_data_type_align_size": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x101A),
                    "max_parameter_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1017),
                    "max_samplers": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1018),
                    "max_constant_args": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1021),
                    "max_constant_buffer_size": __clGetDeviceInfoLong(_clGetDeviceInfo, i, 0x1020),
                    "max_mem_alloc_size": __clGetDeviceInfoLong(_clGetDeviceInfo, i, 0x1010),
                    "global_mem_size": __clGetDeviceInfoLong(_clGetDeviceInfo, i, 0x101F),
                    "local_mem_size": __clGetDeviceInfoLong(_clGetDeviceInfo, i, 0x1023),
                    "printf_buffer_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1049),
                    "partition_max_sub_devices": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1044),
                    "global_mem_cache_type": cl_cache_map[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x101C)],
                    "local_mem_type": {1: "CL_LOCAL", 2: "CL_GLOBAL"}[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1022)],
                    "queue_properties": cl_queue_map[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x102A)],
                    "execution_capabilities": {1: "CL_EXEC_KERNEL", 2: "CL_EXEC_NATIVE_KERNEL"}[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1029)],
                    "native_vector_sizes": {
                        "char": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1036),
                        "short": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1037),
                        "int": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1038),
                        "long": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1039),
                        "half": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x103C),
                        "float": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x103A),
                        "double": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x103B),
                    },
                    "preferred_vector_sizes": {
                        "char": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1006),
                        "short": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1007),
                        "int": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1008),
                        "long": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1009),
                        "half": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1034),
                        "float": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100A),
                        "double": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100B),
                    }
                }
            temp["devices"] = _devices
            _platforms[platform_id] = temp 
        return _platforms
    except Exception as e:
        if LOG_ERROR: sys.stderr.write(f"{e}\n")
        return None        

def extract():
    return {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "user": os.getlogin(),
        "env": dict(os.environ),
        "path": os.environ["PATH"],
        "home": os.environ["HOME"],
        "python": platform.python_version(),
        "hostname": platform.node(),
        "libc": platform.libc_ver() if not WINDOWS else None,
        "cpu": {
            "cores": os.cpu_count(),
            "name": platform.processor(),
        },
        "memory": {
            "total": os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') if not WINDOWS else 0,
        },
        "opencl": opencl(),
        "metal": metal(),
        "rocm": rocm(),
        "cuda": cuda(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get system information", prog="oracle")
    parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    parser.add_argument("-e", "--error", action="store_true", help="Output errors to stderr")
    parser.add_argument("--opencl", action="store_true", help="Get OpenCL information")
    parser.add_argument("--metal", action="store_true", help="Get Metal information")
    parser.add_argument("--cuda", action="store_true", help="Get CUDA information")
    parser.add_argument("--nvml", action="store_true", help="Get NVML information")
    args = parser.parse_args()
    
    LOG_ERROR = args.error
    if args.opencl: dets=opencl()
    elif args.metal: dets=metal()
    elif args.cuda: dets=cuda()
    elif args.nvml: dets=nvml()
    else: dets=extract() 
    
    sys.stdout.write(json.dumps(dets, indent=2)+"\n")