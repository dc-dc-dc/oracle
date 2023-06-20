#include <cuda_runtime.h>
#include <stdio.h>

void main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);
}
