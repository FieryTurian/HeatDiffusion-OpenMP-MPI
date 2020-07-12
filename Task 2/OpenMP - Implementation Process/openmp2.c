#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define N 10000000   // length of the vectors
#define EPS 0.1      // convergence criterium
#define HEAT 100.0   // heat value on the boundary

/*
 timer code from the slides provided
*/
double my_wtime() {
   double retval;
   struct timeval t;
   
   gettimeofday(&t, NULL);
   retval = ((double)t.tv_sec) + ((double)t.tv_usec) / 1000000.0;
   
   return retval;
}

/*
 allocate a vector of length "n"
*/
double *allocVector(int n)
{
   double *v;
   v = (double *)calloc(n, sizeof(double));
   if (v == NULL)
      exit(-1);
   v[0] = HEAT;
   return v;
}

/*
 print the values of a given vector "out" of length "n"
*/
void print(double *in, int n)
{
   int i;

   printf("<");
   for(i=0; i<n; i++) {
      printf(" %f", in[i]);
   }
   printf(">\n");
}

/*
 individual step of the 3-point stencil
 computes values in vector "out" from those in vector "in"
 assuming both are of length "n"
*/
void relax(double *in, double *out, int n)
{
   int i;
   
   for(i=1; i<n-1; i++) {
      out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1];
   }
}

/*
 checks the convergence criterion:
 true, iff for all indices i, we have |out[i] - in[i]| <= eps
*/
bool isStable(double *old, double *new, int n, double eps)
{
   int i;
   bool res = true;

   #pragma omp parallel for schedule(guided,10) reduction (&& : res)
   for(i=1; i<n-1; i++) {
      res = res && (fabs(old[i] - new[i]) <= eps);
   }
   return res;
}

int main()
{
   double *a,*b, *tmp;
   int n = N;
   int iterations = 0;

   a = allocVector(n);
   b = allocVector(n);

   printf("size   : %d M (%d MB)\n", n/1000000, (int)(n*sizeof(double) / (1024*1024)));
   printf("heat   : %f\n", HEAT);
   printf("epsilon: %f\n", EPS);

   omp_set_num_threads(4);

   //Start wallclock time
   double starting = my_wtime();

   do {
      tmp = a;
      a = b;
      b = tmp;
      relax(a, b, n);
      // print(b, n);
      iterations ++;
   } while(!isStable(a, b, n, EPS));

   //End wallclock time
   double stopping = my_wtime();

   free(a);
   free(b);

   printf("Number of iterations: %d\n", iterations);
   printf("Total time: %f seconds\n", (stopping - starting));

   return 0;
}





