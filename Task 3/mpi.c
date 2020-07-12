#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

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
double *allocVector(int n, int rank_id)
{
   double *v;
   v = (double *)calloc(n, sizeof(double));
   if (v == NULL)
      exit(-1);
   if (rank_id == 0) {
      v[1] = HEAT;
   }
   return v;
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
*/
void relax(double *in, double *out, int start, int stop)
{
   for(int i=start; i<=stop; ++i) {
      out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1];
   }
}

/*
 checks the convergence criterion:
 true, iff for all indices i, we have |out[i] - in[i]| <= eps
*/
int isStable(double *old, double *new, int n, double eps)
{
   for(int i=1; i<=n; ++i) {
      if (fabs(old[i] - new[i]) > eps) 
         return false;
   }
   return true;
}

int main(int argc, char *argv[])
{
   //From here on, MPI functions can be used
   MPI_Init(&argc, &argv);

   //Declare variables
   double *a,*b,*tmp;
   int n = N;
   int iterations = 0;
   int rank_id, num_ranks, chunk, last_rank, stop, remainder;
   int start = 1;
   int running = 1;
   int check = 0;
   double starting = 0;
   double stopping = 0;

   //Give processes a rank_id and determine how many ranks are used in this program
   MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

   //Let rank 0 print size, heat and epsilon information
   if (rank_id == 0) {
      printf("size   : %d M (%d MB)\n", n/1000000, (int)(n*sizeof(double) / (1024*1024)));
      printf("heat   : %f\n", HEAT);
      printf("epsilon: %f\n", EPS);
   }

   //If not more than 1 rank is used, we throw an error: we want to run this program in parallel
   if (num_ranks < 2) {
      printf("Please use more than one process\n");
      MPI_Finalize();
      exit(0);
   }

   //Determine how much of the vector each rank should handle
   chunk = n/num_ranks;
   remainder = n % num_ranks;

   //If we have a remainder for chunk: divide over ranks
   if (remainder != 0 && rank_id < remainder) {
      chunk++;
   }

   //Give the last rank a name for reference
   last_rank = num_ranks - 1;

   //Normally every process should stop at chunk in the relax function (and start at 1)
   stop = chunk;
 
   //Rank 0 should start with index 2 in the relax function
   if (rank_id == 0) {
      start = 2;
   }
   
   //The last rank should stop at chunk-1 in the relax function
   if (rank_id == last_rank) {
      stop = chunk - 1;
   }

   //Calloc a piece of memory for every process
   a = allocVector(chunk + 2, rank_id);
   b = allocVector(chunk + 2, rank_id);

   //Start wallclock time
   if (rank_id == 0) {
      starting = my_wtime();
   }

   do {
      //Swap the pointers a and b
      tmp = a;
      a = b;
      b = tmp;

      relax(a, b, start, stop);

      // All neighbors receive data from processes below and above
      if (rank_id != last_rank) {
         // Send last element to next process
         MPI_Send(&b[chunk], 1, MPI_DOUBLE, rank_id + 1, 0, MPI_COMM_WORLD);
      }
      if (rank_id != 0) {
         // Send first element to previous process
         MPI_Send(&b[1], 1, MPI_DOUBLE, rank_id - 1, 0, MPI_COMM_WORLD);
      }
      if (rank_id != last_rank) {
         // Receive first element from next process
         MPI_Recv(&b[chunk+1], 1, MPI_DOUBLE, rank_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (rank_id != 0) {
         // Receive last element from previous process
         MPI_Recv(&b[0], 1, MPI_DOUBLE, rank_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      //If your subvector is stable, set running to 0
      if (isStable(a, b, chunk, EPS)) {
         running = 0;
      }

      //Communicate to all other ranks whether they should stop or not (we stop when check is 0)
      MPI_Allreduce(&running, &check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      iterations++;

   } while(check);

   //End wallclock time and print results
   if (rank_id == 0) {
      stopping = my_wtime();
      printf("Number of iterations: %d\n", iterations);
      printf("Total time: %f seconds\n", (stopping - starting));
   }

   //Free the calloced memory when done
   free(a);
   free(b);

   //Finalize the MPI section
   MPI_Finalize();

   return 0;
}





