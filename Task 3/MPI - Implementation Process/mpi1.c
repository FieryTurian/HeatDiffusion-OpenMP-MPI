#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

//#define N 10000000
#define N 11
#define EPS 0.1
#define HEAT 100.0

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
   v = (double *)malloc(n*sizeof(double));
   return v;
}

void init(double *out, int n)
{
   int i;

   for(i=0; i<n; i++) {
      out[i] = 0;
   }
   out[0] = HEAT;
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

void relax(double *in, double *out, int start, int stop, int before, int after)
{
   int i;
   for(i=start; i<stop; i++) {
      if (i == 0)
         out[i] = 0.25*before + 0.5*in[i] + 0.25*in[i+1];
      else if (i == stop-1)
         out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*after;
      else 
         out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1];
   }
}

bool isStable(double *old, double *new, int n, double eps)
{
   int i;
   bool res=true;

   for(i=1; i<n-1; i++) {
      res = res && (fabs(old[i] - new[i]) <= eps);
   }
   return res;
}

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   double *a,*b,*tmp;
   int n = N;
   int iterations = 0;
   int rank_id, num_ranks, chunk, local_start, last_rank, stop, remainder;
   double recv_above = 0.0;
   double recv_below = 0.0;
   int start = 0;
   int stopping_condition = 0;
   double starting = 0;
   double stopping = 0;

   a = allocVector(n);
   b = allocVector(n);

   init(a, n);
   init(b, n);

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
   if (remainder != 0) {
      if (rank_id < remainder)
         chunk++;
   }

   local_start = rank_id * chunk;
   last_rank = num_ranks - 1;
   stop = chunk;

   if (rank_id == 0) {
      start = 1;
   }
   if (rank_id == last_rank) {
      stop = chunk - 1;
   }
  
   a = a + local_start;
   b = b + local_start;

   //Start wallclock time
   if (rank_id == 0) {
      starting = my_wtime();
   }

   do {
      if (iterations != 0) {
         if (rank_id != 0)
            MPI_Recv(&stopping_condition, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

         if (stopping_condition)
            break;

         tmp = a;
         a = b;
         b = tmp;
      }

      relax(a, b, start, stop, recv_below, recv_above);

      if (num_ranks > 1) {
         // All neighbors receive data from processes below and above
         if (rank_id % 2 == 0) {
            if (rank_id != last_rank) {
               // Send last element to next process
               MPI_Send(&b[chunk-1], 1, MPI_DOUBLE, rank_id + 1, 0, MPI_COMM_WORLD);
               MPI_Recv(&recv_above, 1, MPI_DOUBLE, rank_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank_id != 0) {
               // Send first element to previous process
               MPI_Send(&b[0], 1, MPI_DOUBLE, rank_id - 1, 0, MPI_COMM_WORLD);
               MPI_Recv(&recv_below, 1, MPI_DOUBLE, rank_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
         }
         else {
            // Receive element from previous process
            MPI_Recv(&recv_below, 1, MPI_DOUBLE, rank_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&b[0], 1, MPI_DOUBLE, rank_id - 1, 0, MPI_COMM_WORLD);

            if (rank_id != last_rank) {
               // Receive element from next process
               MPI_Recv(&recv_above, 1, MPI_DOUBLE, rank_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
               MPI_Send(&b[chunk-1], 1, MPI_DOUBLE, rank_id + 1, 0, MPI_COMM_WORLD);
            }  
         }

         if (rank_id == 0) {
            // Put every value calculated by other processes at the right spot in b: gather everything in 0
            for (int i = 1; i < num_ranks; i++) {
               MPI_Recv(b + i * chunk, chunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (isStable(a, b, n, EPS)) {
               stopping_condition = 1;
               for (int i = 1; i < num_ranks; i++) {
                  MPI_Send(&stopping_condition, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
               }
            }
            else {
               for (int i = 1; i < num_ranks; i++) {
                  MPI_Send(&stopping_condition, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
               }
            }
         }
         else {
            MPI_Send(b, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
         }
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
      //print(b, n);
      printf("Number of iterations: %d\n", iterations);
      printf("Total time: %f seconds\n", (stopping - starting));
   }

   MPI_Finalize();
   return 0;
}





