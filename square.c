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

#define DATA_SIZE 10240000

const char *KernelSource =                 "\n"
  "__kernel void square(                    \n"
  "   __global float* input,                \n"
  "   __global float* output,               \n"
  "   const unsigned int count)             \n"
  "{                                        \n"
  "   int i = get_global_id(0);             \n"
  "     output[i] = input[i] * input[i];    \n"
  "}                                        \n"
  "\n";


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
    results[i] = data[i] * data[i];

  TIMERwc_time( &stopsec, &stopnsec);

  printTimeElapsed( "kernel equivalent on host");
}


int main (int argc, char * argv[])
{
  cl_int err;
  cl_kernel kernel;
  size_t global[1];
  size_t local[1];

  if( argc <2) {
    local[0] = 32;
  } else {
    local[0] = atoi(argv[1]);
  }


  printf( "warp size: %d\n", (int)local[0]);

  /* Create data for the run.  */
  float *data = NULL;                /* Original data set given to device.  */
  float *results = NULL;             /* Results returned from device.  */
  int correct;                       /* Number of correct results returned.  */

  int count = DATA_SIZE;
  global[0] = count;

  data = (float *) malloc (count * sizeof (float));
  results = (float *) malloc (count * sizeof (float));

  /* Fill the vector with random float values.  */
  for (int i = 0; i < count; i++)
    data[i] = rand () / (float) RAND_MAX;


  TIMERwc_time( &startsec, &startnsec);

  if( argc > 2) {
    printf( "using openCL on host!\n");
    err = initCPU();
  } else  {
    printf( "using openCL on GPU!\n");
    err = initGPU();
  }
  
  if( err == CL_SUCCESS) {
    kernel = setupKernel( KernelSource, "square", 3, FloatArr, count, data,
                                                     FloatArr, count, results,
                                                     IntConst, count);

    TIMERwc_time( &stopsec, &stopnsec);
    printTimeElapsed( "setup time on host (wallclock)");

    runKernel( kernel, 1, global, local);
  
    TIMERwc_time( &stopsec, &stopnsec);

    printKernelTime();
    printTimeElapsed( "overall wallclock time spent");

    /* Validate our results.  */
    correct = 0;
    for (int i = 0; i < count; i++)
      if (results[i] == data[i] * data[i])
        correct++;

    /* Print a brief summary detailing the results.  */
    printf ("Computed %d/%d %2.0f%% correct values\n", correct, count,
            ((float)correct/(float)count)*100.f);

    err = clReleaseKernel (kernel);
    err = freeDevice();

    timeDirectImplementation( count, data, results);
    
  }


  return 0;
}


