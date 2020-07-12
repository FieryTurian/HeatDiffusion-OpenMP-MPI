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
void relax(double *in, double *out, int start, int end)
{
   int i;

   for(i=start; i<end; i++) {
      out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1];
   }
}

/*
 checks the convergence criterion:
 true, iff for all indices i, we have |out[i] - in[i]| <= eps
*/
bool isStable(double *old, double *new, int start, int end, double eps)
{
   int i;

   for(i=start; i<end; i++) {
      if (fabs(old[i] - new[i]) > eps) 
         return false;
   }
   return true;
}

int main()
{
   double *a, *b, *tmp;
   int n = N;
   int iterations = 0;
   int num_threads = 4;
   bool res = false;

   a = allocVector(n);
   b = allocVector(n);

   printf("size   : %d M (%d MB)\n", n/1000000, (int)(n*sizeof(double) / (1024*1024)));
   printf("heat   : %f\n", HEAT);
   printf("epsilon: %f\n", EPS);

   omp_set_num_threads(num_threads);

   //Start wallclock time
   double starting = my_wtime();

   #pragma omp parallel shared(iterations, res)
   {
      int nr_threads = omp_get_num_threads();
      bool local_res; 
      double local_iterations; 
      int my_id, loop_start, loop_end, i; 

      my_id = omp_get_thread_num();  

      loop_start = my_id * (n/nr_threads);
      if (my_id == 0) 
         loop_start = 1;

      loop_end = (my_id + 1) * (n/nr_threads); 
      if (my_id == nr_threads-1) 
         loop_end = n-1;

      for (local_iterations = 0; !res; ++local_iterations) { 
         /* compute new values */ 
         relax(a, b, loop_start, loop_end);

         /* check for convergence */ 
         local_res = true; 
         #pragma omp barrier
         #pragma omp single
         { 
            res = true;
         } 
      		
         local_res = isStable(a, b, loop_start, loop_end, EPS);

         #pragma omp critical 
         {
            res = res && local_res;
         }

         /* "copy" a to b by swapping pointers */
         #pragma omp barrier
         #pragma omp single
         { 
            tmp = a; 
            a = b; 
            b = tmp; 
         } 
      } 

      /* save number of iterations in shared variable */ 
      #pragma omp single 
      { 
         iterations = local_iterations;
      } 
   }

   //End wallclock time
   double stopping = my_wtime();

   free(a);
   free(b);

   printf("Number of iterations: %d\n", iterations);
   printf("Total time: %f seconds\n", (stopping - starting));

   return 0;
}





