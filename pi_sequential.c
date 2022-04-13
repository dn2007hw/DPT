#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
  int num_steps = 10000;
  double step, pi, sum;
  step = 1.0 / num_steps;
  double x;
  int i;
  double start = omp_get_wtime();

  {
    sum = 0;

    for (i = 0; i < num_steps; i++)
    {
      x = (i + 0.5) * step;
      double aux = 4.0 / (1.0 + x * x);
      sum += aux;
    }
  }
  pi = sum * step;

  double stop = omp_get_wtime();
  double exectime = stop - start;

  printf("PI approximate = %lf\n", pi);
  printf("Exec time = %1f\n", exectime);
}
