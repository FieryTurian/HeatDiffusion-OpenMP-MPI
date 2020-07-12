#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

//#define N 100000    // length of the vectors
//#define EPS 0.0005  // convergence criterium
//#define HEAT 400.0  // heat value on the boundary

#define N 10000000   // length of the vectors
#define EPS 0.01      // convergence criterium
#define HEAT 150.0   // heat value on the boundary

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
 initialise the values of the given vector "out" of length "n"
*/
void init(double *out, int n)
{
   int i;

   for(i=1; i<n; ++i) {
      out[i] = 0;
   }
   out[0] = HEAT;
}

/*
 print the values of a given vector "out" of length "n"
*/
void print(double *in, int n)
{
   int i;

   printf("<");
   for(i=0; i<n; ++i) {
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
   for(i=1; i<n-1; ++i) {
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

   for(i=1; i<n-1; ++i) {
      if (fabs(old[i] - new[i]) > eps) 
         return false;
   }
   return true;
}

int main()
{
   //Declare variables
   double *a,*b, *tmp;
   int n = N;
   int iterations = 0;

   //Calloc a piece of memory for every process
   a = allocVector(n);
   b = allocVector(n);

   //Print size, heat and epsilon information
   printf("size   : %d M (%d MB)\n", n/1000000, (int)(n*sizeof(double) / (1024*1024)));
   printf("heat   : %f\n", HEAT);
   printf("epsilon: %f\n", EPS);

   //Start wallclock time
   double starting = my_wtime();

   do {
      //Swap the pointers a and b
      tmp = a;
      a = b;
      b = tmp;
      relax(a, b, n);
      iterations ++;
   } while(!isStable(a, b, n, EPS));

   //End wallclock time
   double stopping = my_wtime();

   //Free the calloced memory when done
   free(a);
   free(b);

   //Print results
   printf("Number of iterations: %d\n", iterations);
   printf("Total time: %f seconds\n", (stopping - starting));

   return 0;
}





