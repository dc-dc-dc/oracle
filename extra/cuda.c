#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_ERROR_WRAPPER(x) do { cudaError_t err = x; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)
#define NVML_ERROR_WRAPPER(x) do { nvmlReturn_t err = x; if (err != NVML_SUCCESS) { printf("NVML error: %s\n", nvmlErrorString(err)); goto NVML_ERROR; } } while (0)

void print_props(int device_id) {
    struct cudaDeviceProp prop;
    CUDA_ERROR_WRAPPER(cudaGetDeviceProperties(&prop, device_id));
    printf("Device %d \n", device_id);
    printf("\tName: %s\n", prop.name);
    printf("\tUUID: %s\n", prop.uuid.bytes);
    printf("\tLUID: %s\n", prop.luid);
    printf("\tLUID Device Mask: %u\n", prop.luidDeviceNodeMask);
    printf("\tTotal Global Mem: %zu\n", prop.totalGlobalMem);
    printf("\tShared Mem Per Block: %zu\n", prop.sharedMemPerBlock);
    printf("\tRegs Per Block: %d\n", prop.regsPerBlock);
    printf("\tWarp Size: %d\n", prop.warpSize);
    printf("\tMem Pitch: %zu\n", prop.memPitch);
    printf("\tMax Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("\tMax Threads Dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("\tMax Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\tClock Rate: %d\n", prop.clockRate);
    printf("\tTotal Const Mem: %zu\n", prop.totalConstMem);
    printf("\tMajor: %d\n", prop.major);
    printf("\tMinor: %d\n", prop.minor);
    printf("\tTexture Alignment: %zu\n", prop.textureAlignment);
    printf("\tTexture Pitch Alignment: %zu\n", prop.texturePitchAlignment);
    printf("\tDevice Overlap: %d\n", prop.deviceOverlap);
    printf("\tMulti Processor Count: %d\n", prop.multiProcessorCount);
    printf("\tKernel Exec Timeout Enabled: %d\n", prop.kernelExecTimeoutEnabled);
    printf("\tIntegrated: %d\n", prop.integrated);
    printf("\tCan Map Host Memory: %d\n", prop.canMapHostMemory);
    printf("\tCompute Mode: %d\n", prop.computeMode);
    printf("\tMax Texture 1D: %d\n", prop.maxTexture1D);
    printf("\tMax Texture 1D Mipmap: %d\n", prop.maxTexture1DMipmap);
    printf("\tMax Texture 1D Linear: %d\n", prop.maxTexture1DLinear);
    printf("\tMax Texture 2D: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("\tMax Texture 2D Mipmap: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
    printf("\tMax Texture 2D Linear: (%d, %d, %d)\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);
    printf("\tMax Texture 2D Gather: (%d, %d)\n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);
    printf("\tMax Texture 3D: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("\tMax Texture 3D Alt: (%d, %d, %d)\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]);
    printf("\tMax Texture Cubemap: %d\n", prop.maxTextureCubemap);
    printf("\tMax Texture 1D Layered: (%d, %d)\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
    printf("\tMax Texture 2D Layered: (%d, %d, %d)\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
    printf("\tMax Texture Cubemap Layered: (%d, %d)\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);
    printf("\tMax Surface 1D: %d\n", prop.maxSurface1D);
    printf("\tMax Surface 2D: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]);
    printf("\tMax Surface 3D: (%d, %d, %d)\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);
    printf("\tMax Surface 1D Layered: (%d, %d)\n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);
    printf("\tMax Surface 2D Layered: (%d, %d, %d)\n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);
    printf("\tMax Surface Cubemap: %d\n", prop.maxSurfaceCubemap);
    printf("\tMax Surface Cubemap Layered: (%d, %d)\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);
    printf("\tSurface Alignment: %zu\n", prop.surfaceAlignment);
    printf("\tConcurrent Kernels: %d\n", prop.concurrentKernels);
    printf("\tECC Enabled: %d\n", prop.ECCEnabled);
    printf("\tPCI Bus ID: %d\n", prop.pciBusID);
    printf("\tPCI Device ID: %d\n", prop.pciDeviceID);
    printf("\tPCI Domain ID: %d\n", prop.pciDomainID);
    printf("\tTCC Driver: %d\n", prop.tccDriver);
    printf("\tAsync Engine Count: %d\n", prop.asyncEngineCount);
    printf("\tUnified Addressing: %d\n", prop.unifiedAddressing);
    printf("\tMemory Clock Rate: %d\n", prop.memoryClockRate);
    printf("\tMemory Bus Width: %d\n", prop.memoryBusWidth);
    printf("\tL2 Cache Size: %d\n", prop.l2CacheSize);
    printf("\tPersisting L2 Cache Max Size: %d\n", prop.persistingL2CacheMaxSize);
    printf("\tMax Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("\tStream Priorities Supported: %d\n", prop.streamPrioritiesSupported);
    printf("\tGlobal L1 Cache Supported: %d\n", prop.globalL1CacheSupported);
    printf("\tLocal L1 Cache Supported: %d\n", prop.localL1CacheSupported);
    printf("\tShared Mem Per Multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("\tRegisters Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("\tManaged Memory: %d\n", prop.managedMemory);
    printf("\tIs Multi GPU Board: %d\n", prop.isMultiGpuBoard);
    printf("\tMulti GPU Board Group ID: %d\n", prop.multiGpuBoardGroupID);
    printf("\tHost Native Atomic Supported: %d\n", prop.hostNativeAtomicSupported);
    printf("\tSingle To Double Precision Perf Ratio: %d\n", prop.singleToDoublePrecisionPerfRatio);
    printf("\tPageable Memory Access: %d\n", prop.pageableMemoryAccess);
    printf("\tConcurrent Managed Access: %d\n", prop.concurrentManagedAccess);
    printf("\tCompute Preemption Supported: %d\n", prop.computePreemptionSupported);
    printf("\tCan Use Host Pointer For Registered Mem: %d\n", prop.canUseHostPointerForRegisteredMem);
    printf("\tCooperative Launch: %d\n", prop.cooperativeLaunch);
    printf("\tCooperative Multi Device Launch: %d\n", prop.cooperativeMultiDeviceLaunch);
    printf("\tShared Mem Per Block Optin: %zu\n", prop.sharedMemPerBlockOptin);
    printf("\tPageable Memory Access Uses Host Page Tables: %d\n", prop.pageableMemoryAccessUsesHostPageTables);
    printf("\tDirect Managed Mem Access From Host: %d\n", prop.directManagedMemAccessFromHost);
    printf("\tMax Blocks Per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("\tAccess Policy Max Window Size: %d\n", prop.accessPolicyMaxWindowSize);
    printf("\tReserved Shared Mem Per Block: %zu\n", prop.reservedSharedMemPerBlock);
    printf("\tHost Register Supported: %d\n", prop.hostRegisterSupported);
    printf("\tSparse CUDA Array Supported: %d\n", prop.sparseCudaArraySupported);
    printf("\tHost Register Readonly Supported: %d\n", prop.hostRegisterReadOnlySupported);
    printf("\tTimeline Semaphores Supported: %d\n", prop.timelineSemaphoreInteropSupported);
    printf("\tMemory Pools Supported: %d\n", prop.memoryPoolsSupported);
    printf("\tGPUDirect RDMA Supported: %d\n", prop.gpuDirectRDMASupported);
    printf("\tGPUDirect RDMA Flush Writes Options: %d\n", prop.gpuDirectRDMAFlushWritesOptions);
    printf("\tGPUDirect RDMA Writes Ordering: %d\n", prop.gpuDirectRDMAWritesOrdering);
    printf("\tMemory Pool Supported Handle Types: %d\n", prop.memoryPoolSupportedHandleTypes);
    printf("\tDeferred Mapping Cuda Array Supported: %d\n", prop.deferredMappingCudaArraySupported);
    printf("\tIPC Event Supported: %d\n", prop.ipcEventSupported);
    printf("\tCluster Launch: %d\n", prop.clusterLaunch);
    printf("\tUnified Function Pointers: %d\n", prop.unifiedFunctionPointers);
}
int main(int argc, char **argv) {
    int deviceCount;
    CUDA_ERROR_WRAPPER(cudaGetDeviceCount(&deviceCount));
    printf("CUDA Device count: %d\n", deviceCount);
    for(size_t i = 0; i < deviceCount; i++) {
        print_props(i);
    }
    NVML_ERROR_WRAPPER(nvmlInit_v2());
    // Get gpu utilization here
    unsigned int _deviceCount;
    NVML_ERROR_WRAPPER(nvmlDeviceGetCount(&_deviceCount));
    printf("NVML Device count: %d\n", _deviceCount);
    for(size_t i = 0; i < _deviceCount; i++) {
        nvmlDevice_t _device;
        NVML_ERROR_WRAPPER(nvmlDeviceGetHandleByIndex(i, &_device));
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        NVML_ERROR_WRAPPER(nvmlDeviceGetName(_device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
        printf("Device %zu: %s\n", i, name);
        nvmlUtilization_t _utilization;
        NVML_ERROR_WRAPPER(nvmlDeviceGetUtilizationRates(_device, &_utilization));
        printf("Utilization: \n");
        printf("\tGPU:    %d%%\n", _utilization.gpu);
        printf("\tMemory: %d%%\n", _utilization.memory);
        nvmlMemory_t _memory;
        NVML_ERROR_WRAPPER(nvmlDeviceGetMemoryInfo(_device, &_memory));
        printf("Memory: \n");
        printf("\tTotal: %zuGB\n", _memory.total / (1024*1024*1024));
        printf("\tFree:  %zuGB\n", _memory.free / (1024*1024*1024));
        printf("\tUsed:  %zuGB\n", _memory.used / (1024*1024*1024));

    }   
    
    
    NVML_ERROR_WRAPPER(nvmlShutdown());
    return 0;

NVML_ERROR:
    nvmlReturn_t result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    return 1;
}
