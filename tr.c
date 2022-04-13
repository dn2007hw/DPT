// TotientRance.c - Sequential Euler Totient Function (C Version)
// compile: gcc -Wall -O -o TotientRange TotientRange.c
// run:     ./TotientRange lower_num uppper_num

// This program calculates the sum of the totients between a lower and an 
// upper limit using C longs. It is based on earlier work by:

#include <stdio.h>
#include <time.h>

int main(int argc, char ** argv)
{
    int i,j,x,y,t;
    long lower, upper;
    long sum, length;

    sscanf(argv[1], "%ld", &lower);
    sscanf(argv[2], "%ld", &upper);

    sum = 0;
    
    for (i = lower; i <= upper; i++)
    {
        length = 0;
        for (j = 1; j < i; j++)
        {
            t=0;
            x=i;
            y=j;
            {
                while (y != 0) {
                    t = x % y;
                    x = y;
                    y = t;
                }
            }
            if (x==1)
                length++;
        }
        sum = sum + length;
    }

    
    printf("C: Sum of Totients  between [%ld..%ld] is %ld\n",
         lower, upper, sum);

  return 0;
}
