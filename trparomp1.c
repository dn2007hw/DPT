// F21DP CW1 OpenMP parallelism implementation by Daya Natarajan
//
// TotientRance.c - Sequential Euler Totient Function (C Version)
// compile: gcc -Wall -O -o TotientRange TotientRange.c
// run:     ./TotientRange lower_num uppper_num

// Greg Michaelson 14/10/2003
// Patrick Maier   29/01/2010 [enforced ANSI C compliance]

// This program calculates the sum of the totients between a lower and an 
// upper limit using C longs. It is based on earlier work by:
// Phil Trinder, Nathan Charles, Hans-Wolfgang Loidl and Colin Runciman

#include <stdio.h>
#include <time.h>
#include <omp.h>

// hcf x 0 = x
// hcf x y = hcf y (rem x y)

long hcf(long x, long y)
{
  long t;

  while (y != 0) {
    t = x % y;
    x = y;
    y = t;
  }
  return x;
}


// relprime x y = hcf x y == 1

int relprime(long x, long y)
{
  return hcf(x, y) == 1;
}


// euler n = length (filter (relprime n) [1 .. n-1])

long euler(long n)
{
  long length, i;

  length = 0;
  for (i = 1; i < n; i++)
    if (relprime(n, i))
      length++;

  return length;
}


// sumTotient lower upper = sum (map euler [lower, lower+1 .. upper])
// F21DP CW1 OpenMP parallelism implementation by Daya Natarajan
long sumTotient(long lower, long upper)
{
  long sum[100000], total;
  int i,num_t; 	

#pragma omp parallel
{
  	int i, id, num_threads;
	id = omp_get_thread_num();
	num_threads = omp_get_num_threads();
	printf("No of threads within parallel :%d and thread no:%d \n",num_threads,id);

	if(id==0)
    		num_t = num_threads;

  	sum[id] = 0;
 	for(i=id+lower; i<upper; i+=num_threads) {
  		sum[id] += euler(i);
	}


}

	total=0;
	for (i=0;i<num_t;i++)
		total+=sum[i];

  return total;
}


void runBenchmark()
{
  clock_t start, end;
  double time_taken;

  for (long i = 1; i < 1000000 ; i = i + 100000) {
    start = clock();
    euler(i);
    end = clock();
    time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("euler(%lu) = %f seconds\n", i, time_taken);
  }   
}

int main(int argc, char ** argv)
{
  long lower, upper;

  if (argc != 3) {
    printf("not 2 arguments\n");
    return 1;
  }
//  int tnum = atoi(argv[3]);	
//  omp_set_num_threads(10);

  sscanf(argv[1], "%ld", &lower);
  sscanf(argv[2], "%ld", &upper);

  printf("There are %d of threads in the sequential region. \n", omp_get_num_threads());
  printf("There are maximum %d of threads in the system. \n", omp_get_max_threads());
  printf("There are %d of processors in the system. \n", omp_get_num_procs());
  double start = omp_get_wtime();
  long sum=0; 
	
	printf("There are %d threads in the parallel region and thread no %d. \n", omp_get_num_threads(),omp_get_thread_num());

	sum = sumTotient(lower, upper);

	printf("C: Sum of Totients  between [%ld..%ld] is %ld\n",
         lower, upper, sum);

	double stop = omp_get_wtime();
 	double exectime = stop - start;
  	printf("Execution time: %1f\n", exectime);
  return 0;
}

