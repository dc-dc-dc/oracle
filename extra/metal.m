#import <Metal//Metal.h>
#import <Foundation/Foundation.h>

int main(int argc, char** argv) {
    id <MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("no default device found");
    }
    NSLog(@"name: %@", device.name);
    NSLog(@"maxThreadgroupMemoryLength: %lu", device.maxThreadgroupMemoryLength);
    NSLog(@"maxThreadsPerThreadgroup: [%lu, %lu, %lu]", device.maxThreadsPerThreadgroup.width, device.maxThreadsPerThreadgroup.height, device.maxThreadsPerThreadgroup.depth);
    NSLog(@"supportsRaytracing: %@", device.supportsRaytracing ? @"Yes" : @"No");
    NSLog(@"supportsPrimitiveMotionBlur: %@", device.supportsPrimitiveMotionBlur ? @"Yes" : @"No");
    NSLog(@"supportsRaytracingFromRender: %@", device.supportsRaytracingFromRender ? @"Yes" : @"No");
    NSLog(@"supports32BitMSAA: %@", device.supports32BitMSAA ? @"Yes" : @"No");
    NSLog(@"supportsPullModelInterpolation: %@", device.supportsPullModelInterpolation ? @"Yes" : @"No");
    NSLog(@"areRasterOrderGroupsSupported: %@", device.areRasterOrderGroupsSupported ? @"Yes" : @"No");
    NSLog(@"supportsShaderBarycentricCoordinates: %@", device.supportsShaderBarycentricCoordinates ? @"Yes" : @"No");
    NSLog(@"areProgrammableSamplePositionsSupported: %@", device.areProgrammableSamplePositionsSupported ? @"Yes" : @"No");
    NSLog(@"supportsBCTextureCompression: %@", device.supportsBCTextureCompression ? @"Yes" : @"No");
    NSLog(@"isDepth24Stencil8PixelFormatSupported: %@", device.isDepth24Stencil8PixelFormatSupported ? @"Yes" : @"No");
    NSLog(@"supportsQueryTextureLOD: %@", device.supportsQueryTextureLOD ? @"Yes" : @"No");
    NSLog(@"supports32BitFloatFiltering: %@", device.supports32BitFloatFiltering ? @"Yes" : @"No");
    NSLog(@"supportsBCTextureCompression: %@", device.supportsBCTextureCompression ? @"Yes" : @"No");
    NSLog(@"supportsFunctionPointers: %@", device.supportsFunctionPointers ? @"Yes" : @"No");
    NSLog(@"supportsFunctionPointersFromRender: %@", device.supportsFunctionPointersFromRender ? @"Yes" : @"No");
    NSLog(@"currentAllocatedSize: %lu", device.currentAllocatedSize);
    NSLog(@"recommendedMaxWorkingSetSize: %llu", device.recommendedMaxWorkingSetSize);
    NSLog(@"hasUnifiedMemory: %@", device.hasUnifiedMemory ? @"Yes" : @"No");
    NSLog(@"maxTransferRate: %llu", device.maxTransferRate);

    return 0;
}