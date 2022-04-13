#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef OSX
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "timer.h"
#include "math.h"
#include "simple.h"

#define DATA_SIZE 1024

const char *KernelSource =                 "\n"
  "__kernel void matmul(                    \n"
  "   __global float* in_a,                 \n"
  "   __global float* in_b,                 \n"
  "   __global float* out,                  \n"
  "   const unsigned int count)             \n"
  "{                                        \n"
  "   int i = get_global_id(0);             \n"
  "   int j = get_global_id(1);             \n"
  "   float sum=0.0;                        \n"
  "   for( int k=0; k< count; k++) {        \n"
  "     sum += in_a[i*count+k] *in_b[k*count+j]; \n"
  "   }                                     \n"
  "   out[i*count+j] = sum;                 \n"
  "}                                        \n"
  "\n";


int startsec, startnsec, stopsec, stopnsec;

void printTimeElapsed( char *text)
{
  double elapsed = (stopsec -startsec)*1000.0
                  + (double)(stopnsec -startnsec)/1000000.0;
  printf( "%s: %f msec\n", text, elapsed);
}

void timeDirectImplementation( int count, float* in_a, float* in_b, float *out)
{
  float sum;

  TIMERwc_time( &startsec, &startnsec);

  for (int i = 0; i < count; i++) {
    for (int j = 0; j < count; j++) {
      sum = 0.0;
      for (int k = 0; k < count; k++) {
        sum += in_a[i*count+k] * in_b[k*count+j];
      }
      out[i*count+j] =sum;
    }
  }
  TIMERwc_time( &stopsec, &stopnsec);

  printTimeElapsed( "kernel equivalent on host");
}


int main (int argc, char * argv[])
{
  cl_int err;
  cl_kernel kernel;
  size_t global[2];
  size_t local[2];

  if( argc <2) {
    local[0] = 16;
    local[1] = 16;
  } else {
    local[0] = atoi(argv[1]);
    local[1] = atoi(argv[2]);
  }

  printf( "warp size: %d, %d\n", (int)local[0], (int)local[1]);

  /* Create data for the run.  */
  float *in_a = NULL;                /* Original data set given to device.  */
  float *in_b = NULL;                /* Original data set given to device.  */
  float *out = NULL;             /* Results returned from device.  */
  int correct;                       /* Number of correct results returned.  */

  int count = DATA_SIZE;
  float sum;
  global[0] = count;
  global[1] = count;

  in_a = (float *) malloc (count * count * sizeof (float));
  in_b = (float *) malloc (count * count * sizeof (float));
  out = (float *) malloc (count * count * sizeof (float));

  /* Fill the vector with random float values.  */
  for (int i = 0; i < count*count; i++) {
    in_a[i] = rand () / (float) RAND_MAX;
    in_b[i] = rand () / (float) RAND_MAX;
  }

  TIMERwc_time( &startsec, &startnsec);

  if( argc > 3) {
    printf( "using openCL on host!\n");
    err = initCPU();
  } else  {
    printf( "using openCL on GPU!\n");
    err = initGPU();
  }
  
  if( err == CL_SUCCESS) {
    kernel = setupKernel( KernelSource, "matmul", 4, FloatArr, count*count, in_a,
                                                     FloatArr, count*count, in_b,
                                                     FloatArr, count*count, out,
                                                     IntConst, count);
    TIMERwc_time( &stopsec, &stopnsec);
    printTimeElapsed( "setup time on host (wallclock)");

    runKernel( kernel, 2, global, local);
  
    TIMERwc_time( &stopsec, &stopnsec);

    printKernelTime();
    printTimeElapsed( "overall wallclock time spent");

    /* Validate our results.  */
    correct = 0;
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < count; j++) {
        sum = 0.0;
        for (int k = 0; k < count; k++) {
          sum += in_a[i*count+k] * in_b[k*count+j];
        }
        if ( fabsf(out[i*count+j] - sum) < 0.0001)
          correct++;
      }
    }

    /* Print a brief summary detailing the results.  */
    printf ("Computed %d/%d %2.0f%% correct values\n", correct, count*count,
            ((float)correct/(count*count))*100.f);

    err = clReleaseKernel (kernel);
    err = freeDevice();

    timeDirectImplementation( count, in_a, in_b, out);
    
  }


  return 0;
}


