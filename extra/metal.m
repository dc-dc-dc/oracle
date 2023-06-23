#import <Metal//Metal.h>
#import <Foundation/Foundation.h>

int main(int argc, char** argv) {
    id <MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("no default device found");
    }
    NSLog(@"name: %@", device.name);
    NSLog(@"areProgrammableSamplePositionsSupported: %@", device.areProgrammableSamplePositionsSupported ? @"Yes" : @"No");
    NSLog(@"areRasterOrderGroupsSupported: %@", device.areRasterOrderGroupsSupported ? @"Yes" : @"No");
    NSLog(@"currentAllocatedSize: %lu", device.currentAllocatedSize);
    NSLog(@"hasUnifiedMemory: %@", device.hasUnifiedMemory ? @"Yes" : @"No");
    NSLog(@"isDepth24Stencil8PixelFormatSupported: %@", device.isDepth24Stencil8PixelFormatSupported ? @"Yes" : @"No");
    NSLog(@"isHeadless: %@", device.isHeadless ? @"Yes" : @"No");
    NSLog(@"isLowPower: %@", device.isLowPower ? @"Yes" : @"No");
    NSLog(@"isRemovable: %@", device.isRemovable ? @"Yes" : @"No");
    NSLog(@"location: %lu", device.location);
    NSLog(@"locationNumber: %lu", device.locationNumber);
    NSLog(@"maxThreadgroupMemoryLength: %lu", device.maxThreadgroupMemoryLength);
    NSLog(@"maxTransferRate: %llu", device.maxTransferRate);
    NSLog(@"maxThreadsPerThreadgroup: [%lu, %lu, %lu]", device.maxThreadsPerThreadgroup.width, device.maxThreadsPerThreadgroup.height, device.maxThreadsPerThreadgroup.depth);
    NSLog(@"peerGroupID: %llu", device.peerGroupID);
    NSLog(@"peerCount: %i", device.peerCount);
    NSLog(@"peerIndex: %i", device.peerIndex);
    NSLog(@"recommendedMaxWorkingSetSize: %llu", device.recommendedMaxWorkingSetSize);
    NSLog(@"registryID: %llu", device.registryID);
    NSLog(@"supportsBCTextureCompression: %@", device.supportsBCTextureCompression ? @"Yes" : @"No");
    NSLog(@"supportsFunctionPointers: %@", device.supportsFunctionPointers ? @"Yes" : @"No");
    NSLog(@"supportsFunctionPointersFromRender: %@", device.supportsFunctionPointersFromRender ? @"Yes" : @"No");
    NSLog(@"supportsPrimitiveMotionBlur: %@", device.supportsPrimitiveMotionBlur ? @"Yes" : @"No");
    NSLog(@"supportsQueryTextureLOD: %@", device.supportsQueryTextureLOD ? @"Yes" : @"No");
    NSLog(@"supportsPullModelInterpolation: %@", device.supportsPullModelInterpolation ? @"Yes" : @"No");
    NSLog(@"supportsRaytracing: %@", device.supportsRaytracing ? @"Yes" : @"No");
    NSLog(@"supportsRaytracingFromRender: %@", device.supportsRaytracingFromRender ? @"Yes" : @"No");
    NSLog(@"supportsShaderBarycentricCoordinates: %@", device.supportsShaderBarycentricCoordinates ? @"Yes" : @"No");
    NSLog(@"supports32BitFloatFiltering: %@", device.supports32BitFloatFiltering ? @"Yes" : @"No");
    NSLog(@"supports32BitMSAA: %@", device.supports32BitMSAA ? @"Yes" : @"No");

    return 0;
}