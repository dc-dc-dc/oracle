#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ERROR_WRAPPER(x) do { cudaError_t err = x; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } } while (0)

int main(int argc, char **argv) {
    int deviceCount;
    ERROR_WRAPPER(cudaGetDeviceCount(&deviceCount));
    printf("Device count: %d\n", deviceCount);
    for(size_t i = 0; i < deviceCount; i++) {
        struct cudaDeviceProp prop;
        ERROR_WRAPPER(cudaGetDeviceProperties(&prop, i));
        printf("Device %zu \n", i);
        printf("\tName: %s\n", prop.name);
    }
    return 0;
}
