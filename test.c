#include <OpenCL/opencl.h>
#include <stdio.h>

#define HANDLE(f)                     \
    {                                 \
        if (f != CL_SUCCESS)          \
        {                             \
            printf("Error: %d\n", f); \
            return 1;                 \
        }                             \
    }

int main(int argc, char *argv[])
{
    uint platform_count;
    HANDLE(clGetPlatformIDs(0, NULL, &platform_count));
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
    HANDLE(clGetPlatformIDs(platform_count, platforms, NULL));
    printf("Platform count: %d\n", platform_count);
    for (int i = 0; i < platform_count; i++)
    {
        printf("Platform %d:\n", i);
        char *name;
        size_t name_size;
        printf("ID: %p\n", platforms[i]);
        HANDLE(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &name_size));
        name = (char *)malloc(sizeof(char) * name_size);
        HANDLE(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_size, name, NULL));
        printf("Name: %s\n", name);
    }
    return 0;
}