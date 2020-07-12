#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define N 10000000
//#define N 10
#define EPS 0.1
#define HEAT 10000.0

double my_wtime() {
   double retval;
   struct timeval t;
   
   gettimeofday(&t, NULL);
   retval = ((double)t.tv_sec) + ((double)t.tv_usec) / 1000000.0;
   
   return retval;
}

double *allocVector(int n)
{
   double *v;
   v = (double *)calloc(n, sizeof(double));
   if (v == NULL)
      exit(-1);
   v[0] = HEAT;
   return v;
}

void init(double *out, int n)
{
   int i;

   for(i=1; i<n; ++i) {
      out[i] = 0;
   }
   out[0] = HEAT;
}

void print(double *in, int n)
{
   int i;

   printf("<");
   for(i=0; i<n; ++i) {
      printf(" %f", in[i]);
   }
   printf(">\n");
}

void relax(double *in, double *out, int n)
{
   int i;
   for(i=1; i<n-1; ++i) {
      out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1];
   }
}

bool isStable(double *old, double *new, int n, double eps)
{
   int i;

   for(i=1; i<n-1; ++i) {
      if (fabs(old[i] - new[i]) > eps) 
         return false;
   }
   return true;
}

int main()
{
   double *a,*b, *tmp;
   int n = N;
   int iterations = 0;
   int bound = 2;

   a = allocVector(n);
   b = allocVector(n);

   //init(a, N);
   //init(b, N);

   //n = N;

   printf("size   : %d M (%d MB)\n", n/1000000, (int)(n*sizeof(double) / (1024*1024)));
   printf("heat   : %f\n", HEAT);
   printf("epsilon: %f\n", EPS);

//Start wallclock time
   double starting = my_wtime();

   do {
      if (bound < n)
         bound ++;
      tmp = a;
      a = b;
      b = tmp;
      relax(a, b, bound);
      //print(b, n);
      iterations ++;
   } while(!isStable(a, b, bound, EPS));

//End wallclock time
   double stopping = my_wtime();

   printf("printing first few 100 elements instead of all %d elements:\n", n);
   print(b, 100);

   free(a);
   free(b);

   printf("Number of iterations: %d\n", iterations);
   printf("Total time: %f seconds\n", (stopping - starting));

   return 0;
}





