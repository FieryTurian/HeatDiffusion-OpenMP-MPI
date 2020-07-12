#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define N 10000000
//#define N 13
#define EPS 0.1
#define HEAT 100.0

double my_wtime() {
   double retval;
   struct timeval t;
   
   gettimeofday(&t, NULL);
   retval = ((double)t.tv_sec) + ((double)t.tv_usec) / 1000000.0;
   
   return retval;
}

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

void print(double *in, int n)
{
   int i;

   printf("<");
   for(i=0; i<n; i++) {
      printf(" %f", in[i]);
   }
   printf(">\n");
}

void relax(double *in, double *out, int start, int stop)
{
   for(int i=start; i<=stop; i++) {
      out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1];
   }
}

int isStable(double *old, double *new, int n, double eps)
{
   for(int i=1; i<n-1; ++i) {
      if (fabs(old[i] - new[i]) > eps) 
         return false;
   }
   return true;
}

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   double *a,*b,*tmp;
   int n = N;
   int iterations = 0;
   int rank_id, num_ranks, chunk, last_rank, stop, remainder;
   int start = 1;
   int running = 1;
   int check = 1;
   double starting = 0;
   double stopping = 0;

   MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

   if (rank_id == 0) {
      printf("size   : %d M (%d MB)\n", n/1000000, (int)(n*sizeof(double) / (1024*1024)));
      printf("heat   : %f\n", HEAT);
      printf("epsilon: %f\n", EPS);
   }

   chunk = n/num_ranks;
   remainder = n % num_ranks;

   //If we have a remainder for chunk: divide over processes
   if (remainder != 0 && rank_id < remainder) {
      chunk++;
   }

   last_rank = num_ranks - 1;
   stop = chunk;

   if (rank_id == 0) {
      start = 2;
   }
   if (rank_id == last_rank) {
      stop = chunk - 1;
   }

   a = allocVector(chunk + 2, rank_id);
   b = allocVector(chunk + 2, rank_id);

   //Start wallclock time
   if (rank_id == 0) {
      starting = my_wtime();
   }

   do {
      if (!check)
         break;

      tmp = a;
      a = b;
      b = tmp;

      relax(a, b, start, stop);

      if (num_ranks > 1) {
         // All neighbors receive data from processes below and above
         if (rank_id % 2 == 0) {
            if (rank_id != last_rank) {
               // Send last element to next process
               MPI_Send(&b[chunk], 1, MPI_DOUBLE, rank_id + 1, 0, MPI_COMM_WORLD);
               // Receive first element from next process
               MPI_Recv(&b[chunk+1], 1, MPI_DOUBLE, rank_id + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank_id != 0) {
               // Send first element to previous process
               MPI_Send(&b[1], 1, MPI_DOUBLE, rank_id - 1, 2, MPI_COMM_WORLD);
               // Receive last element from previous process
               MPI_Recv(&b[0], 1, MPI_DOUBLE, rank_id - 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
         }
         else {
            // Receive last element from previous process
            MPI_Recv(&b[0], 1, MPI_DOUBLE, rank_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Send first element to previous process
            MPI_Send(&b[1], 1, MPI_DOUBLE, rank_id - 1, 1, MPI_COMM_WORLD);

            if (rank_id != last_rank) {
               // Receive first element from next process
               MPI_Recv(&b[chunk+1], 1, MPI_DOUBLE, rank_id + 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
               // Send last element to next process
               MPI_Send(&b[chunk], 1, MPI_DOUBLE, rank_id + 1, 3, MPI_COMM_WORLD);
            }  
         }

         if (isStable(a, b, chunk, EPS)) {
            running = 0;
         }
         MPI_Allreduce(&running, &check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      }
      else {
         printf("Please use more than one process\n");
         MPI_Finalize();
         exit(0);
      }

      iterations++;
   } while(true);

   //End wallclock time and print results
   if (rank_id == 0) {
      stopping = my_wtime();
      printf("Number of iterations: %d\n", iterations);
      printf("Total time: %f seconds\n", (stopping - starting));
   }

   MPI_Finalize();
   return 0;
}





