////////////////////////////////////////////////////////////////////////////////
//
// Guo, Jackson, Jiang, and Pape
// Professor Annexstein
// Parallel Computing
// 19 April 2011
// Lab 2: Sieve of Eratosthenes
//
////////////////////////////////////////////////////////////////////////////////


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <eratosthenes_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"

#include "eratosthenes.h"

void timetest(int);

int SieveParallelBasic(int);
int SieveParallelFirst(int);
int SieveParallelSecond(int);
int SieveParallelOrig(int);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
   if(argc == 2){
      timetest(atoi(argv[1]));
   }else{
      for(int i = 16; i <= 30; i++){
         timetest(i);
      } // end for
   } // end if
} // end main

void timetest(int pwr){
   // initialize variables and read input args?
   time_t startt, endt;
   int primes;

   int num = pow(2,pwr);
   printf("Finding primes less than %d (2^%d):\n", num, pwr);

   // compute and time linear? (should take 10 minutes for 2^30)
   time(&startt);
   primes = eratosLinear(num, 0);
   time(&endt);
   printf("Found %d primes using the linear function; ", primes);
   printf("took %d seconds.\n", int(endt-startt));

   
   // compute and time basic parallel
   time(&startt);
   primes = SieveParallelBasic(pwr);
   time(&endt);
   printf("Found %d primes using the basic parallel function; ", primes);
   printf("took %d seconds.\n", int(endt-startt));
 
   //// compute and time first imprevement
   //time(&startt);
   //primes = SieveParallelFirst(pwr);
   //time(&endt);
   //printf("Found %d primes using the first improvement; ", primes);
   //printf("took %d seconds.\n", int(endt-startt));
 
   //// compute and time second imprevement
   //time(&startt);
   //primes = SieveParallelSecond(pwr);
   //time(&endt);
   //printf("Found %d primes using the second improvement; ", primes);
   //printf("took %d seconds.\n", int(endt-startt));
 

   // compute and time 3 improvements

} // end timetest


////////////////////////////////////////////////////////////////////////////////
//! Basic parallel algorithm
// @param pwr: the power of two for which to complete the Sieve
// @return: the number of primes between 0 and 2^pwr - 1
////////////////////////////////////////////////////////////////////////////////
int SieveParallelBasic(int pwr)
{
//printf("Begin SieveParallelBasic\n");
   int num = pow(2,pwr);
   char *sieve = (char*)malloc(num*sizeof(char));
   if(!sieve){return -1;}
   char* devsieve;
   cudaMalloc((void**)&devsieve,num*sizeof(char));
   if(!devsieve){ free(sieve); return -1; }
//printf("Sieve Mallocced: %d bytes\n", int(sizeof(sieve)));

  
   // initialize the sieve to 1's
   for(int i = 0; i < num; i++)
   {
      sieve[i] = 1;
   } // end for
//printf("Sieve init'd\n");
   
   sieve[0] = sieve[1] = 0; // 0 and 1 are not considered prime
   
   int k = 2; // 2 is the first prime
   // k*k is the first composite number that may not have been marked as
   // composite (since k*(k-1) also has k-1 as a factor, it has been marked)
   while(k*k <= num)
   {
      //int mult = k*k;
      //while(mult <= num)
      //{
      //   sieve[mult] = 0;
      //   mult += k;
      //} // end while
      cudaMemcpy(devsieve, sieve, num*sizeof(char), cudaMemcpyHostToDevice);
      int gridx = int(ceil(float(num)/float(k-1)/512.0));
//printf("grid dim for %d: %d\n", k, gridx);
      dim3 grid(gridx,1,1);
      dim3 threads(512,1,1);
      ThreadIsMult<<<grid,threads>>>(k, devsieve, num);
      cudaMemcpy(sieve, devsieve, num*sizeof(char), cudaMemcpyDeviceToHost);
//for(int i = 0; i < 50 && k < 30; i++){printf("%d,",int(sieve[i]));} printf("\n");
      k = findsmallest(sieve, num, k+1);
   } // end while
   return sumLinear(sieve, num);
} // end SieveParallelBasic

//////////////////////////////////////////////////////////////////////////////////
////! Improved parallel algorithm
//// @param pwr: the power of two for which to complete the Sieve
//// @return: the number of primes between 0 and 2^pwr - 1
//////////////////////////////////////////////////////////////////////////////////
//int SieveParallelImproved(int pwr)
//{
//   // calculate primes thru 2^15 linearly (sqrt(2^30))
//   // place primes in an int array "primes"
//   // for each 16KB section of sieve
//      //call kernel:
//         // initialize section of sieve to 1's
//         // for each prime in "primes"
//            // set all multiples to 0 in section of sieve
//      // copy section of sieve from device to host memory
//   // sum sieve to get the number of primes
//} // end SieveParallelImproved      

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// code beyond this point has been shown to cause errors
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//! First Improvement over the Basic parallel algorithm
// @param pwr: the power of two for which to complete the Sieve
// @return: the number of primes between 0 and 2^pwr - 1
////////////////////////////////////////////////////////////////////////////////
int SieveParallelFirst(int pwr)
{
   printf("Begin SieveParallelFirst\n");
   int num = pow(2,pwr);
   char *sieve = (char*)malloc(num*sizeof(char));
   if(!sieve){return -1;}
   char* devsieve;
   cudaMalloc((void**)&devsieve,num*sizeof(char));
   if(!devsieve){ free(sieve); return -1; }
  
   // initialize the sieve to 1's
   for(int i = 0; i < num; i++)
   {
      sieve[i] = 1;
   } // end for
   printf("Sieve init'd\n");
   
   sieve[0] = sieve[1] = 0; // 0 and 1 are not considered prime
   
   int k = 2; // 2 is the first prime
   // k*k is the first composite number that may not have been marked as
   // composite (since k*(k-1) also has k-1 as a factor, it has been marked)
   while(k*k <= num)
   {
      cudaMemcpy(devsieve, sieve, num*sizeof(char), cudaMemcpyHostToDevice);
      dim3 grid(4,1,1);
      dim3 threads(128,1,1);
      ThreadIsMult1<<<grid,threads>>>(k, devsieve, num);
      cudaMemcpy(sieve, devsieve, num*sizeof(char), cudaMemcpyDeviceToHost);
   for(int i = 0; i < 50 && k < 30; i++){printf("%d,",int(sieve[i]));} printf("\n");
      k = findsmallest(sieve, num, k+1);
   } // end while
   return sumLinear(sieve, num);
} // end SieveParallelFirst

////////////////////////////////////////////////////////////////////////////////
//! Second Improvement over the Basic parallel algorithm
// @param pwr: the power of two for which to complete the Sieve
// @return: the number of primes between 0 and 2^pwr - 1
////////////////////////////////////////////////////////////////////////////////
int SieveParallelSecond(int pwr)
{
   printf("Begin SieveParallelSecond\n");
   int num = pow(2,pwr);
   char *sieve = (char*)malloc(num*sizeof(char));
   if(!sieve){return -1;}
   char* devsieve;
   cudaMalloc((void**)&devsieve,num*sizeof(char));
   if(!devsieve){ free(sieve); return -1; }
  
   // initialize the sieve to 1's
   for(int i = 0; i < num; i++){sieve[i] = 1;} // end for
   
   //printf("Sieve init'd\n");
   
   sieve[0] = sieve[1] = 0; // 0 and 1 are not considered prime
   
   //essemble the grid
   dim3 grid(4,1,1);
   dim3 threads(512,1,1);
   
   //printf("Grid init'd\n");
   
   int k = 2;
   while(k*k <= num){
      
      // obtain an array of primes
      int primesl = 0;
      for(int i = k; i < k*k; i++)
         if(sieve[i]==1){primesl++;}
      int *primes = (int*)malloc(primesl*sizeof(int));
      if(!primes){printf("Cut out in primes init!"); free(sieve); return -1;}
      
      //gather the prime numbers
      int i = 0; int j = k;
      while(j < k*k){
         if(sieve[j]==1){primes[i]=j; i++;}
         j++;
      }//end loop
      
      if(int *temp = (int*)malloc(primesl*sizeof(int))){
         for(int x = 0; x < primesl; x++){temp[x] = primes[x];}
         primes = temp;
      }//end if
      
      printf("Primes init'd, %d %d %d\n", k, j, primesl);
      for(int x = 0; x < primesl; x++){printf("%d,", int(primes[x]));}printf("\n");
      
      int *devprimes;
      cudaMalloc((void**)&devprimes, primesl*sizeof(int));
      if(!devprimes){ free(sieve); free(primes); return -1; }
      
      cudaMemcpy(devsieve, sieve, num*sizeof(char), cudaMemcpyHostToDevice);
      cudaMemcpy(devprimes, primes, primesl*sizeof(int), cudaMemcpyHostToDevice);
      ThreadIsMult2<<<grid,threads>>>(devprimes, primesl, devsieve, num);
      cudaMemcpy(sieve, devsieve, num*sizeof(char), cudaMemcpyDeviceToHost);
      cudaFree(devprimes);
      free(primes);
      //for(int x = 0; x < primesl; x++){printf("%d", int(sieve[x]));}printf("\n\n\n\n");
      
      k = findsmallest(sieve, num, (k*k)+1);
   }//end loop

   return sumLinear(sieve, num);
} // end SieveParallelSecond

////////////////////////////////////////////////////////////////////////////////
//! Original parallel algorithm
// @param pwr: the power of two for which to complete the Sieve
// @return: the number of primes between 0 and 2^pwr - 1
////////////////////////////////////////////////////////////////////////////////
int SieveParallelOrig(int pwr)
{
printf("beginning original function\n");
   int num = pow(2,pwr); //number of primes
   // make sure we have sufficient num to justify parallelization
   if( num < SH_MEM_SIZE )
   {
printf("linear sol :(\n");
      return eratosLinear(num, 0);
   }
   // initialize variables
   int lnum = min(num, SH_MEM_SIZE); // number of primes to find linearly
   int i = 0; // i is the number of primes whose multiples have been set to 0
   int j; // j is the number of primes we have found so far (primes[i]^2)
   int *devprimes, *primes; // int array of primes found so far
   char *devsieve, *sieve; // char array representing the Sieve of Eratosthenes
                           // (0 indicates a composite number)
   int gridx; // dimension of the grid
   cudaMalloc((void**)&devsieve, num);
   if(!devsieve){ return -1; }
   sieve = (char*)malloc(lnum);
   if(!sieve){ cudaFree(devsieve); return -1; }
//printf("sieve mallocced\n");

   // initialize sieve on device, then copy the first section to the host.
   gridx = num/SH_MEM_SIZE; // each block is responsible for SH_MEM_SIZE data.
                              // Create enough blocks to cover enire sieve.
printf("gridx %d\n",gridx);
   dim3 grid(gridx,1,1);
   dim3 threads(512,1,1);
   InitSieve<<<grid,threads>>>(devsieve, num);
printf("sieve initd\n");
////////////////////////////////////////////////////////////////////////////////
//for(int n = 0; n < num; n++)
//{
//   sieve[n] = 1;
//} // end for
//cudaMemcpy(devsieve, sieve, num, cudaMemcpyHostToDevice);
////////////////////////////////////////////////////////////////////////////////
   cudaMemcpy(sieve, devsieve, lnum, cudaMemcpyDeviceToHost);
//for(int n = 0; n < lnum; n++){ if(!(sieve[n]==1)){ printf("%dth position %d!\n", n, int(sieve[n])); }}
for(int n = 0; n < 40; n++){ printf("%d,", int(sieve[n])); }

   //calculate primes through 4096 linearly ~>512 primes
   j = eratosLinear(lnum, sieve);
printf("%d primes found below %d\n", j, lnum);
   cudaMemcpy(devsieve, sieve, lnum, cudaMemcpyHostToDevice);


   //
   // Find the first set of primes
   //
   
   // int primessize ensures that our primes array fits in an integer of blocks
   int primessize = ((j*num/lnum -1)/(SH_MEM_INT) + 1 +1)*SH_MEM_SIZE; ///////////////////this is a waste of memory
   cudaMalloc((void**)&devprimes, primessize);
   if(!devprimes){ cudaFree(devsieve); free(sieve); return -1; }
   primes = (int*)malloc(j*sizeof(int)*num/lnum);
   if(!primes){ cudaFree(devsieve);free(sieve);cudaFree(devprimes); return -1; }
//printf("primes malloccd: %dbytes\n", primessize);

   // initialize primes to 0
   gridx = primessize/SH_MEM_SIZE;
   grid = dim3(gridx,1,1);
   InitPrimes<<<grid,threads>>>(devprimes, primessize/sizeof(int));
   cudaMemcpy(primes, devprimes, primessize, cudaMemcpyDeviceToHost);
////////////////////////////////////////////////////////////////////////////////
//for( int n = 0; n < primessize/sizeof(int); n++ )
//{
//   primes[n] = 0;
//} // end for
//cudaMemcpy(devprimes, primes, primessize, cudaMemcpyHostToDevice);
////////////////////////////////////////////////////////////////////////////////
//for(int n=0; n<20; n++){printf("%d,",primes[n]);}
//printf("\n");

   
   // find first set of primes
   primes[0] = findsmallest(sieve, num, 0);
   for(int l = 1; l < j; l++)
   {
      primes[l] = findsmallest(sieve, num, primes[l-1]+1);
   } // end for
   j--; // decrement, because we use j as an index in 0-indexed arrays
   cudaMemcpy(devprimes, primes, j*sizeof(int), cudaMemcpyHostToDevice);
for(int n=0; n<20; n++){printf("%d,",primes[n]);}
   
   //mark (remove multiples of) 512 primes in parallel:
   //set up blocks
   gridx = num/SH_MEM_SIZE; // one block for each 16KB of memory
   grid = dim3(gridx,1,1);
   //remove multiples of lower primes (found using linear algorithm) from sieve
   ExtendSieve<<<grid,threads>>>(j, devprimes, devsieve, num);
   i = j;
for(int n = i-2; n < i+2; n++){ printf("%d, ", primes[n]); }
printf("sieve extended to %d\n",i);
printf("\n");

   
   //
   // remove larger primes from the sieve
   while( primes[i]*primes[i] <= num ) // while there are primes left to mark
   {
      //copy global sieve to host
      cudaMemcpy(sieve+primes[i], devsieve+primes[i],
                  primes[i]*primes[i] - primes[i], cudaMemcpyDeviceToHost);
//printf("section of sieve copied to host\n");
      //// find new primes
      //while primes[j] < primes[i]**2
      // anything less than primes[i]^2 and not marked composite must be prime;
      // however, we will most likely jump past this limit and find a
      // possibly composite number.  So, after the loop to find primes, we must
      // remove the last "prime" just in case.
      int oldj = j;
      while( primes[j] < pow(primes[i],2)  &&  primes[j] != -1 )
      {
         j++;
         primes[j] = findsmallest(sieve, num, primes[j-1]+1);
      } // end while
      primes[j--] = 0;
printf("highest prime found so far: %d\n", primes[j]);
      cudaMemcpy(devprimes+oldj, primes+oldj, j-oldj, cudaMemcpyHostToDevice);
//printf("section of primes copied to device\n");
      
      //call kernel with new primes
      gridx = max(1,(j-i+1)/512);
      grid = dim3(gridx,1,1); // enough blocks to mark most newly-
                              // found primes, or at least 1 block
      // stay with 512 threads in a line for now
      SieveRemove<<<grid,threads>>>(i, devprimes, j, devsieve, num);
      i = min(j, i + gridx*512); // we have marked the lesser of all the found
                                 // primes or 512 primes for each block
   } // end while there are primes left to mark

   // free memory
printf("freeing memory:");
   free(devsieve);
   free(sieve);
   free(devprimes);
   free(primes);
printf(" complete\n");
   return j; /// we actually might want to return something else (or parallel sum those primes whose squares are less than our maximum)
} // end SieveParallelOrig

