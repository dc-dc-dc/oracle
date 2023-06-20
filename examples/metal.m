#import <Metal//Metal.h>

int main(int argc, char** argv) {
    id <MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("no default device found");
    }
    NSLog(@"name: %@", device.name);
    return 0;
}