__kernel void calculateTR(__global const float *a,
	__global const float *b,
	__global float *c) {
		
		int gid = get_global_id(0);
		c[gid] = a[gid] + b[gid];
    
    int i,j,x,y,t;
    long lower, upper;
    long sum, length;
    
    lower = a;
    upper = b;
    
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
    
    c = sum;
    
	}
