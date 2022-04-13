#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef OSX
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "timer.h"
#include "simple.h"

#define DATA_SIZE 4096

#ifdef VERSION1

const char *KernelSource =                      "\n"
  "__kernel void transpose(                      \n"
  "   __global float* input,                     \n"
  "   __global float* output,                    \n"
  "   const unsigned int count)                  \n"
  "{                                             \n"
  "   int i = get_global_id(0);                  \n"
  "   int j = get_global_id(1);                  \n"
  "     output[i*count+j] = input[j*count+i];    \n"
  "}                                             \n"
  "\n";

#elif defined VERSION2

const char *KernelSource =                      "\n"
  "__kernel void transpose(                      \n"
  "   __global float* input,                     \n"
  "   __global float* output,                    \n"
  "   const unsigned int count)                  \n"
  "{                                             \n"
  "   int i = get_global_id(0);                  \n"
  "   int j = get_global_id(1);                  \n"
  "     output[j*count+i] = input[i*count+j];    \n"
  "}                                             \n"
  "\n";

#elif defined VERSION3

const char *KernelSource =                                   "\n"
  "__kernel void transpose(                                   \n"
  "   __global float* input,                                  \n"
  "   __global float* output,                                 \n"
  "   const unsigned int count,                               \n"
  "   __local float* shmem)                                   \n"
  "{                                                          \n"
  "   int i = get_global_id(0);                               \n"
  "   int j = get_global_id(1);                               \n"
  "   int li = get_local_id(0);                               \n"
  "   int lj = get_local_id(1);                               \n"
  "   int gi = get_group_id(0);                               \n"
  "   int gj = get_group_id(1);                               \n"
  "   int sz = get_local_size(0);                             \n"
  "                                                           \n"
  "     shmem[li*sz+lj] = input[i*count+j];                   \n"
  "     barrier( CLK_LOCAL_MEM_FENCE);                        \n"
  "     output[(gj*sz+li)*count+(gi*sz+lj)] = shmem[lj*sz+li];\n"
  "}                                                          \n"
  "\n";

#else 

const char *KernelSource =                                   "\n"
  "__kernel void transpose(                                   \n"
  "   __global float* input,                                  \n"
  "   __global float* output,                                 \n"
  "   const unsigned int count,                               \n"
  "   __local float* shmem)                                   \n"
  "{                                                          \n"
  "   int i = get_global_id(0);                               \n"
  "   int j = get_global_id(1);                               \n"
  "   int li = get_local_id(0);                               \n"
  "   int lj = get_local_id(1);                               \n"
  "   int gi = get_group_id(0);                               \n"
  "   int gj = get_group_id(1);                               \n"
  "   int sz = get_local_size(0);                             \n"
  "                                                           \n"
  "     shmem[lj*sz+li] = input[j*count+i];                   \n"
  "     barrier( CLK_LOCAL_MEM_FENCE);                        \n"
  "     output[(gi*sz+lj)*count+(gj*sz+li)] = shmem[li*sz+lj];\n"
  "}                                                          \n"
  "\n";

#endif

#define die(msg, ...) do {                      \
  (void) fprintf (stderr, msg, ## __VA_ARGS__); \
  (void) fprintf (stderr, "\n");                \
} while (0)

int startsec, startnsec, stopsec, stopnsec;

void printTimeElapsed( char *text)
{
  double elapsed = (stopsec -startsec)*1000.0
                  + (double)(stopnsec -startnsec)/1000000.0;
  printf( "%s: %f msec\n", text, elapsed);
}

void timeDirectImplementation( int count, float* data, float* results)
{
  TIMERwc_time( &startsec, &startnsec);

  for (int i = 0; i < count; i++)
    for (int j = 0; j < count; j++)
#if defined VERSION1 || VERSION3
      results[i*count+j] = data[j*count+i];
#else
      results[j*count+i] = data[i*count+j];
#endif

  TIMERwc_time( &stopsec, &stopnsec);

  printTimeElapsed( "kernel equivalent on host");
}


int main (int argc, char * argv[])
{
  cl_int err;
  cl_kernel kernel;
  size_t global[2];
  size_t local[2];

  if( argc <3) {
    local[0] = 32;
    local[1] = 32;
  } else {
    local[0] = atoi(argv[1]);
    local[1] = atoi(argv[2]);
  }

  printf( "work group size: %d, %d\n", (int)local[0], (int)local[1]);

#ifdef VERSION3
  if( local[0] != local[1])
    die( "Error: version 3 requires quadratic workgroups size!");
#elif defined VERSION4
  if( local[0] != local[1])
    die( "Error: version 4 requires quadratic workgroups size!");
#endif

  /* Create data for the run.  */
  float *data = NULL;                /* Original data set given to device.  */
  float *results = NULL;             /* Results returned from device.  */
  int correct;                       /* Number of correct results returned.  */

  int count = DATA_SIZE;
  global[0] = count;
  global[1] = count;

  data = (float *) malloc (count * count * sizeof (float));
  results = (float *) malloc (count * count * sizeof (float));

  /* Fill the vector with random float values.  */
  for (int i = 0; i < count; i++)
    for (int j = 0; j < count; j++)
      data[i*count+j] = rand () / (float) RAND_MAX;


  TIMERwc_time( &startsec, &startnsec);

  if( argc > 3) {
    printf( "using openCL on host!\n");
    err = initCPU();
  } else  {
    printf( "using openCL on GPU!\n");
    err = initGPU();
  }

  if( err == CL_SUCCESS) {
#if defined VERSION3 || defined VERSION4
    kernel = setupKernel( KernelSource, "transpose", 4, FloatArr, count*count, data,
                                                        FloatArr, count*count, results,
                                                        IntConst, count,
                                                        LocalFloat, local[0]*local[1]);
#else
    kernel = setupKernel( KernelSource, "transpose", 3, FloatArr, count*count, data,
                                                        FloatArr, count*count, results,
                                                        IntConst, count);
#endif

    TIMERwc_time( &stopsec, &stopnsec);
    printTimeElapsed( "setup time on host (wallclock)");

    runKernel( kernel, 2, global, local);
  
    TIMERwc_time( &stopsec, &stopnsec);

    printKernelTime();
    printTimeElapsed( "overall wallclock time spent");

    /* Validate our results.  */
    correct = 0;
    for (int i = 0; i < count; i++)
      for (int j = 0; j < count; j++)
        if (results[i*count+j] == data[j*count+i])
          correct++;

    /* Print a brief summary detailing the results.  */
    printf ("Computed %d/%d %2.0f%% correct values\n", correct, count*count,
            ((float)correct/(float)(count*count))*100.f);

    err = clReleaseKernel (kernel);
    err = freeDevice();

    timeDirectImplementation( count, data, results);
    
  }


  return 0;
}


