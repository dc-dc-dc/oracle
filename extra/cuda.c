#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

// print methods
#define PRINTL_STR(name, val) printf("%-50s: %s\n", name, val)
#define PRINTL_INT(name, val) printf("%-50s: %d\n", name, val)
#define PRINTL_2INT(name, val1, val2) printf("%-50s: %d, %d\n", name, val1, val2)
#define PRINTL_3INT(name, val1, val2, val3) printf("%-50s: %d, %d, %d\n", name, val1, val2, val3)

#define PRINTL_UINT(name, val) printf("%-50s: %u\n", name, val)
#define PRINTL_SIZE_T(name, val) printf("%-50s: %zu\n", name, val)

#define CUDA_ERROR_WRAPPER(x) do { cudaError_t err = x; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)
#define NVML_ERROR_WRAPPER(x) do { nvmlReturn_t err = x; if (err != NVML_SUCCESS) { printf("NVML error: %s\n", nvmlErrorString(err));  goto NVML_ERROR; } } while (0)

void print_props(int device_id) {
    struct cudaDeviceProp prop;
    CUDA_ERROR_WRAPPER(cudaGetDeviceProperties(&prop, device_id));
    printf("Device(%d):\n", device_id);
    PRINTL_STR("\tName", prop.name);
    PRINTL_STR("\tUUID", prop.uuid.bytes);
    PRINTL_STR("\tLUID", prop.luid);
    PRINTL_UINT("\tLUID Device Mask", prop.luidDeviceNodeMask);
    PRINTL_SIZE_T("\tTotal Global Mem", prop.totalGlobalMem);
    PRINTL_SIZE_T("\tShared Mem Per Block", prop.sharedMemPerBlock);
    PRINTL_INT("\tRegs Per Block", prop.regsPerBlock);
    PRINTL_INT("\tWarp Size", prop.warpSize);
    PRINTL_SIZE_T("\tMem Pitch", prop.memPitch);
    PRINTL_INT("\tMax Threads Per Block", prop.maxThreadsPerBlock);
    PRINTL_3INT("\tMax Threads Dim", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    PRINTL_3INT("\tMax Grid Size", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    PRINTL_INT("\tClock Rate", prop.clockRate);
    PRINTL_SIZE_T("\tTotal Const Mem", prop.totalConstMem);
    PRINTL_INT("\tMajor", prop.major);
    PRINTL_INT("\tMinor", prop.minor);
    PRINTL_SIZE_T("\tTexture Alignment", prop.textureAlignment);
    PRINTL_SIZE_T("\tTexture Pitch Alignment", prop.texturePitchAlignment);
    PRINTL_INT("\tDevice Overlap", prop.deviceOverlap);
    PRINTL_INT("\tMulti Processor Count", prop.multiProcessorCount);
    PRINTL_INT("\tKernel Exec Timeout Enabled", prop.kernelExecTimeoutEnabled);
    PRINTL_INT("\tIntegrated", prop.integrated);
    PRINTL_INT("\tCan Map Host Memory", prop.canMapHostMemory);
    PRINTL_INT("\tCompute Mode", prop.computeMode);
    PRINTL_INT("\tMax Texture 1D", prop.maxTexture1D);
    PRINTL_INT("\tMax Texture 1D Mipmap", prop.maxTexture1DMipmap);
    PRINTL_INT("\tMax Texture 1D Linear", prop.maxTexture1DLinear);
    PRINTL_2INT("\tMax Texture 2D", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    PRINTL_2INT("\tMax Texture 2D Mipmap", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
    PRINTL_3INT("\tMax Texture 2D Linear", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);
    PRINTL_2INT("\tMax Texture 2D Gather", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);
    PRINTL_3INT("\tMax Texture 3D", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    PRINTL_3INT("\tMax Texture 3D Alt", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]);
    PRINTL_INT("\tMax Texture Cubemap", prop.maxTextureCubemap);
    PRINTL_2INT("\tMax Texture 1D Layered", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
    PRINTL_3INT("\tMax Texture 2D Layered", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
    PRINTL_2INT("\tMax Texture Cubemap Layered", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);
    PRINTL_INT("\tMax Surface 1D", prop.maxSurface1D);
    PRINTL_2INT("\tMax Surface 2D", prop.maxSurface2D[0], prop.maxSurface2D[1]);
    PRINTL_3INT("\tMax Surface 3D", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);
    PRINTL_2INT("\tMax Surface 1D Layered", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);
    PRINTL_3INT("\tMax Surface 2D Layered", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);
    PRINTL_INT("\tMax Surface Cubemap", prop.maxSurfaceCubemap);
    PRINTL_2INT("\tMax Surface Cubemap Layered", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);
    PRINTL_SIZE_T("\tSurface Alignment", prop.surfaceAlignment);
    PRINTL_INT("\tConcurrent Kernels", prop.concurrentKernels);
    PRINTL_INT("\tECC Enabled", prop.ECCEnabled);
    PRINTL_INT("\tPCI Bus ID", prop.pciBusID);
    PRINTL_INT("\tPCI Device ID", prop.pciDeviceID);
    PRINTL_INT("\tPCI Domain ID", prop.pciDomainID);
    PRINTL_INT("\tTCC Driver", prop.tccDriver);
    PRINTL_INT("\tAsync Engine Count", prop.asyncEngineCount);
    PRINTL_INT("\tUnified Addressing", prop.unifiedAddressing);
    PRINTL_INT("\tMemory Clock Rate", prop.memoryClockRate);
    PRINTL_INT("\tMemory Bus Width", prop.memoryBusWidth);
    PRINTL_INT("\tL2 Cache Size", prop.l2CacheSize);
    PRINTL_INT("\tPersisting L2 Cache Max Size", prop.persistingL2CacheMaxSize);
    PRINTL_INT("\tMax Threads Per MultiProcessor", prop.maxThreadsPerMultiProcessor);
    PRINTL_INT("\tStream Priorities Supported", prop.streamPrioritiesSupported);
    PRINTL_INT("\tGlobal L1 Cache Supported", prop.globalL1CacheSupported);
    PRINTL_INT("\tLocal L1 Cache Supported", prop.localL1CacheSupported);
    PRINTL_SIZE_T("\tShared Mem Per Multiprocessor", prop.sharedMemPerMultiprocessor);
    PRINTL_INT("\tRegisters Per Multiprocessor", prop.regsPerMultiprocessor);
    PRINTL_INT("\tManaged Memory", prop.managedMemory);
    PRINTL_INT("\tIs Multi GPU Board", prop.isMultiGpuBoard);
    PRINTL_INT("\tMulti GPU Board Group ID", prop.multiGpuBoardGroupID);
    PRINTL_INT("\tHost Native Atomic Supported", prop.hostNativeAtomicSupported);
    PRINTL_INT("\tSingle To Double Precision Perf Ratio", prop.singleToDoublePrecisionPerfRatio);
    PRINTL_INT("\tPageable Memory Access", prop.pageableMemoryAccess);
    PRINTL_INT("\tConcurrent Managed Access", prop.concurrentManagedAccess);
    PRINTL_INT("\tCompute Preemption Supported", prop.computePreemptionSupported);
    PRINTL_INT("\tCan Use Host Pointer For Registered Mem", prop.canUseHostPointerForRegisteredMem);
    PRINTL_INT("\tCooperative Launch", prop.cooperativeLaunch);
    PRINTL_INT("\tCooperative Multi Device Launch", prop.cooperativeMultiDeviceLaunch);
    PRINTL_SIZE_T("\tShared Mem Per Block Optin", prop.sharedMemPerBlockOptin);
    PRINTL_INT("\tPageable Memory Access Uses Host Page Tables", prop.pageableMemoryAccessUsesHostPageTables);
    PRINTL_INT("\tDirect Managed Mem Access From Host", prop.directManagedMemAccessFromHost);
    PRINTL_INT("\tMax Blocks Per Multiprocessor", prop.maxBlocksPerMultiProcessor);
    PRINTL_INT("\tAccess Policy Max Window Size", prop.accessPolicyMaxWindowSize);
    PRINTL_SIZE_T("\tReserved Shared Mem Per Block", prop.reservedSharedMemPerBlock);
    PRINTL_INT("\tHost Register Supported", prop.hostRegisterSupported);
    PRINTL_INT("\tSparse CUDA Array Supported", prop.sparseCudaArraySupported);
    PRINTL_INT("\tHost Register Readonly Supported", prop.hostRegisterReadOnlySupported);
    PRINTL_INT("\tTimeline Semaphores Supported", prop.timelineSemaphoreInteropSupported);
    PRINTL_INT("\tMemory Pools Supported", prop.memoryPoolsSupported);
    PRINTL_INT("\tGPUDirect RDMA Supported", prop.gpuDirectRDMASupported);
    PRINTL_INT("\tGPUDirect RDMA Flush Writes Options", prop.gpuDirectRDMAFlushWritesOptions);
    PRINTL_INT("\tGPUDirect RDMA Writes Ordering", prop.gpuDirectRDMAWritesOrdering);
    PRINTL_INT("\tMemory Pool Supported Handle Types", prop.memoryPoolSupportedHandleTypes);
    PRINTL_INT("\tDeferred Mapping Cuda Array Supported", prop.deferredMappingCudaArraySupported);
    PRINTL_INT("\tIPC Event Supported", prop.ipcEventSupported); 
    PRINTL_INT("\tCluster Launch", prop.clusterLaunch);
    PRINTL_INT("\tUnified Function Pointers", prop.unifiedFunctionPointers);
}
int main(int argc, char **argv) {
    int deviceCount;
    CUDA_ERROR_WRAPPER(cudaGetDeviceCount(&deviceCount));
    printf("CUDA Devices(%d):\n", deviceCount);
    for(size_t i = 0; i < deviceCount; i++) {
        print_props(i);
    }
    NVML_ERROR_WRAPPER(nvmlInit_v2());
    // Get gpu utilization here
    unsigned int _deviceCount;
    NVML_ERROR_WRAPPER(nvmlDeviceGetCount(&_deviceCount));
    printf("\n");
    printf("NVML Devices(%d):\n", _deviceCount);
    for(size_t i = 0; i < _deviceCount; i++) {
        nvmlDevice_t _device;
        NVML_ERROR_WRAPPER(nvmlDeviceGetHandleByIndex(i, &_device));
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        NVML_ERROR_WRAPPER(nvmlDeviceGetName(_device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
        printf("Device(%zu):\n", i);
        PRINTL_STR("Name", name);
        unsigned int fan_speed;
        NVML_ERROR_WRAPPER(nvmlDeviceGetFanSpeed(_device, &fan_speed));
        PRINTL_INT("Fan Speed(%):", fan_speed);
        nvmlUtilization_t _utilization;
        NVML_ERROR_WRAPPER(nvmlDeviceGetUtilizationRates(_device, &_utilization));
        printf("Utilization(%%): \n");
        PRINTL_INT("\tGPU", _utilization.gpu);
        PRINTL_INT("\tMemory", _utilization.memory);
        nvmlMemory_t _memory;
        NVML_ERROR_WRAPPER(nvmlDeviceGetMemoryInfo(_device, &_memory));
        printf("Memory(GB): \n");
        PRINTL_SIZE_T("\tTotal", _memory.total / (1024*1024*1024));
        PRINTL_SIZE_T("\tFree", _memory.free / (1024*1024*1024));
        PRINTL_SIZE_T("\tUsed", _memory.used / (1024*1024*1024));
        unsigned int pcount;
        nvmlProcessInfo_t pinfo;
        nvmlReturn_t res = nvmlDeviceGetGraphicsRunningProcesses(_device, &pcount, &pinfo);
        if(res == NULL) {
            printf("GOT NULL\n");
            goto NVML_ERROR;
            return 1;
        }
        if(res != 0) {
            NVML_ERROR_WRAPPER(res);
        }
        printf("Processes(%u):\n", pcount);
        // printf("Processes(%u):\n", process_count);
        // for(size_t j = 0; j < *process_count; j++) {
        //     printf("Process(%zu):\n", j);
        //     PRINTL_UINT("\tPID", process_info[j].pid);
        //     PRINTL_SIZE_T("\tUsed Memory", process_info[j].usedGpuMemory);
        // }
    }   
    
    
    NVML_ERROR_WRAPPER(nvmlShutdown());
    return 0;

NVML_ERROR:
    nvmlReturn_t result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    return 1;
}
