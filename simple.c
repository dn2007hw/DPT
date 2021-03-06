#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#ifdef OSX
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "timer.h"
#include "simple.h"

typedef struct {
  clarg_type arg_t;
  cl_mem dev_buf;
  float *host_buf;
  int    num_elems;
  int    val;
} kernel_arg;

#define MAX_ARG 10


#define die(msg, ...) do {                      \
  (void) fprintf (stderr, msg, ## __VA_ARGS__); \
  (void) fprintf (stderr, "\n");                \
} while (0)

/* global setup */

static cl_platform_id cpPlatform;     /* openCL platform.  */
static cl_device_id device_id;        /* Compute device id.  */
static cl_context context;            /* Compute context.  */
static cl_command_queue commands;     /* Compute command queue.  */
static cl_program program;            /* Compute program.  */
static cl_event event_timer;          /* timing info */
static int num_kernel_args;
static kernel_arg kernel_args[MAX_ARG];

static unsigned long startCL, stopCL;
static int starts, startns, stops, stopns;


cl_int initDevice ( int devType)
{
  cl_int err = CL_SUCCESS;
  cl_uint num_platforms;
  cl_platform_id *cpPlatforms;

  /* Connect to a compute device.  */
  err = clGetPlatformIDs (0, NULL, &num_platforms);
  if (CL_SUCCESS != err) {
    die ("Error: Failed to find a platform!");
  } else {
    cpPlatforms = (cl_platform_id *)malloc( sizeof( cl_platform_id)*num_platforms);
    err = clGetPlatformIDs(num_platforms, cpPlatforms, NULL);

    for(uint i=0; i<num_platforms; i++){
        err = clGetDeviceIDs(cpPlatforms[i], devType, 1, &device_id, NULL);
        if (err == CL_SUCCESS ) {
           cpPlatform = cpPlatforms[i];
           break;
        }
    }
    if (CL_SUCCESS != err) {
      die ("Error: Failed to find a platform!");
    } else {
      /* Get a device of the appropriate type.  */
      err = clGetDeviceIDs (cpPlatform, devType, 1, &device_id, NULL);
      if (CL_SUCCESS != err) {
        die ("Error: Failed to create a device group!");
      } else { 
        /* Create a compute context.  */
        context = clCreateContext (0, 1, &device_id, NULL, NULL, &err);
        if (!context || err != CL_SUCCESS) {
          die ("Error: Failed to create a compute context!");
        } else {
          /* Create a command commands.  */
          commands = clCreateCommandQueue (context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
          if (!commands || err != CL_SUCCESS) {
            die ("Error: Failed to create a command commands!");
          }
        }
      }
    }
  }

  return err;
}

cl_int initCPU ()
{
  return initDevice( CL_DEVICE_TYPE_CPU);
}

cl_int initGPU ()
{
  return initDevice( CL_DEVICE_TYPE_GPU);
}

cl_kernel setupKernel( const char *kernel_source, char *kernel_name, int num_args, ...)
{
  cl_kernel kernel = NULL;
  cl_int err = CL_SUCCESS;
  va_list ap;
  int i;
  
  /* Create the compute program from the source buffer.  */
  program = clCreateProgramWithSource (context, 1,
                                       (const char **) &kernel_source,
                                       NULL, &err);
  if (!program || err != CL_SUCCESS) {
    die ("Error: Failed to create compute program!");
  }

  /* Build the program executable.  */
  err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];

      clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG,
                             sizeof (buffer), buffer, &len);
      die ("Error: Failed to build program executable!\n%s", buffer);
    }

  /* Create the compute kernel in the program.  */
  kernel = clCreateKernel (program, kernel_name, &err);
  if (!kernel || err != CL_SUCCESS) {
    die ("Error: Failed to create compute kernel!");
    kernel = NULL;
  } else {

    num_kernel_args = num_args;
    va_start(ap, num_args);
    for(i=0; (i<num_args) && (kernel != NULL); i++) {
      kernel_args[i].arg_t =va_arg(ap, clarg_type);
      switch( kernel_args[i].arg_t) {
        case FloatArr:
          kernel_args[i].num_elems = va_arg(ap, int);
          kernel_args[i].host_buf = va_arg(ap, float *);
          /* Create the device memory vector  */
          kernel_args[i].dev_buf = clCreateBuffer (context, CL_MEM_READ_WRITE,
                                                   sizeof (float) * kernel_args[i].num_elems, NULL, NULL);
          if (!kernel_args[i].dev_buf ) {
            die ("Error: Failed to allocate device memory for arg %d!", i+1);
            kernel = NULL;
          } else {
            err = clEnqueueWriteBuffer( commands, kernel_args[i].dev_buf, CL_TRUE, 0,
                                                  sizeof (float) * kernel_args[i].num_elems,
                                                  kernel_args[i].host_buf, 0, NULL, NULL);
            if( CL_SUCCESS != err) {
              die ("Error: Failed to write to source array for arg %d!", i+1);
              kernel = NULL;
            }
            err = clSetKernelArg (kernel, i, sizeof (cl_mem), &kernel_args[i].dev_buf);
            if( CL_SUCCESS != err) {
              die ("Error: Failed to set kernel arg %d!", i);
              kernel = NULL;
            }
          }
          break;
        case IntConst:
          kernel_args[i].val = va_arg(ap, unsigned int);
          err = clSetKernelArg (kernel, i, sizeof (unsigned int), &kernel_args[i].val);
          if( CL_SUCCESS != err) {
            die ("Error: Failed to set kernel arg %d!", i);
            kernel = NULL;
          }
          break;
        case LocalFloat:
          kernel_args[i].num_elems = va_arg(ap, unsigned int);
          err = clSetKernelArg (kernel, i, kernel_args[i].num_elems * sizeof(float), NULL);
          if( CL_SUCCESS != err) {
            die ("Error: Failed to set kernel arg %d!", i);
            kernel = NULL;
          }
          break;
        default:
          die ("Error: illegal argument tag for executeKernel!");
          kernel = NULL;
      }
    }
  }
  va_end(ap);

  return kernel;
}

cl_int runKernel( cl_kernel kernel, cl_uint dim, size_t *global, size_t *local)
{
  cl_int err;

  TIMERwc_time( &starts, &startns);
  if (CL_SUCCESS
      != clEnqueueNDRangeKernel (commands, kernel,
                                 dim, NULL, global, local, 0, NULL, &event_timer))
    die ("Error: Failed to execute kernel!");


  /* Wait for all commands to complete.  */
  err = clFinish (commands);
  if( CL_SUCCESS != err) {
    if( err == CL_OUT_OF_HOST_MEMORY) {
      die ("Error: clFinish failed: Out of host memory!");
    } else  {
      die ("Error: clFinish failed: invalid command queue!");
    }
  }
      
  TIMERwc_time( &stops, &stopns);
  if( CL_SUCCESS != 
      clGetEventProfilingInfo( event_timer, CL_PROFILING_COMMAND_START,
                               sizeof(cl_ulong), &startCL, NULL)) {
    die("Error: no profiling info for start!");
  }
  if( CL_SUCCESS !=
      clGetEventProfilingInfo( event_timer, CL_PROFILING_COMMAND_END,
                               sizeof(cl_ulong), &stopCL, NULL)) {
    die("Error: no profiling info for end!");
  }

  for( int i=0; i< num_kernel_args; i++) {
    if( kernel_args[i].arg_t == FloatArr) {
      err = clEnqueueReadBuffer (commands, kernel_args[i].dev_buf,
                              CL_TRUE, 0, sizeof (float) * kernel_args[i].num_elems,
                              kernel_args[i].host_buf, 0, NULL, NULL);
      if( err != CL_SUCCESS) 
        die( "Error: Failed to transfer back arg %d!", i);
    }
  }

  return err;
}

void printKernelTime()
{
  double elapsedCL = (stopCL-startCL)/1000000.0;
  printf( "time spent on GPU: %f msec\n", elapsedCL);

  double elapsed = (stops -starts)*1000.0
                  + (stopns -startns)/1000000.0;
  printf( "time spent on kernel: %f msec\n", elapsed);
}

cl_int freeDevice()
{
  cl_int err;

  for( int i=0; i< num_kernel_args; i++) {
    if( kernel_args[i].arg_t == FloatArr) 
      err = clReleaseMemObject (kernel_args[i].dev_buf);
  }
  err = clReleaseProgram (program);
  err = clReleaseCommandQueue (commands);
  err = clReleaseContext (context);

  return err;
}



