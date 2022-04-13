

// Headers for openCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif



#include <stdio.h>
#include <stdlib.h>



int main() {
    // Getting platform count
    cl_uint platCount;
    clGetPlatformIDs(0, NULL, &platCount);
    
    // Allocate memory, get list of platforms
    cl_platform_id *platforms = (cl_platform_id*) malloc(platCount*sizeof(cl_platform_id));
    clGetPlatformIDs(platCount, platforms, NULL);
    
    printf("platform count %d: \n", platCount);
    
    // Iterate over platforms
    cl_uint i;
    for (i=0; i<platCount; ++i) {
        char buf[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL);
        printf("platform %d: vendor     '%s'\n", i, buf);
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
        printf("platform %d: NAME       '%s'\n", i, buf);
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(buf), buf, NULL);
        printf("platform %d: VERSION    '%s'\n", i, buf);
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, sizeof(buf), buf, NULL);
        printf("platform %d: PROFILE    '%s'\n", i, buf);
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, sizeof(buf), buf, NULL);
        printf("platform %d: EXT        '%s'\n", i, buf);
        
        cl_uint devCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &devCount);
        
                printf("device count %d: \n", devCount);
        
        cl_device_id *devices = (cl_device_id*) malloc(devCount*sizeof(cl_device_id));
        
        // List of devices in platform
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devCount, devices, NULL);
        

        
        cl_uint j;
        for (j=0; j<devCount; ++j) {
            char buf[256];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_NAME        : '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_VERSION     : '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(buf), buf, NULL);
            printf("device %d CL_DRIVER_VERSION     : '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_VENDOR      : '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_PROFILE, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_PROFILE     : '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_EXTENSIONS  : '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_MAX_COMPUTE_UNITS: '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_MAX_WORK_GROUP_SIZE: '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_GLOBAL_MEM_SIZE: '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_LOCAL_MEM_SIZE: '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_MAX_WORK_ITEM_SIZES: '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: '%s'\n", j, buf);
            
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf), buf, NULL);
            printf("device %d CL_DEVICE_MAX_CLOCK_FREQUENCY: '%s'\n", j, buf);
        }
        free(devices);
    }
    free(platforms);
    return 0;
}
