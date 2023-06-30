import ctypes, sys
from oracle.util import WINDOWS, error_wrap, getdict, ascii_str, LOG_ERROR


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