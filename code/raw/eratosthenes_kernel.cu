
#ifndef _ERATOSTHENES_KERNEL_H_
#define _ERATOSTHENES_KERNEL_H_

#include <stdio.h>
#include "eratosthenes.h"

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
// Sieve kernel thread specification
__global__ void SieveKernelBasic(/*params*/)
{
}

////////////////////////////////////////////////////////////////////////////////
//!SieveRemove
//!Sieve removal step, remove values in primes
// optimized for removing multiples of large primes: because multiples of these
// primes are few and far between, it would be a waste of time to copy the
// sieve into shared memory and back into global; we simply mark multiples in
// global memory.
////////////////////////////////////////////////////////////////////////////////
__global__ void SieveRemove(int i, int primes[], int pl, char sieve[], int sl)
{
   //get K for the current thread
   int x = i+threadIdx.x+512*blockIdx.x;
   if(x < pl){
      int k = primes[x];
   
      //Remove multiples of k
      //if(k > 1){
         int mult = k*k;
         while(mult < sl){
            sieve[mult]=0;
            mult += k;
         }//end while
      //}// end if
   }//end if
}//end SieveRemove

////////////////////////////////////////////////////////////////////////////////
//! ExtendSieve
//! Extends the Sieve by marking multiples of primes in the existing section
//! of the Sieve in the new section of the Sieve
//  @param j: int: the index of the heighest prime
//  @param primes: int*: array of prime numbers we have found, trailed by 0's
//  @param sieve: char*: the sieve of Eratosthenes (0 indicates composite index)
////////////////////////////////////////////////////////////////////////////////
__global__ void ExtendSieve(int j, int *primes, char *sieve, int sl){
   for(int m = 0; m <= j; m++){
      int k = primes[m];
      int nextSect = SH_MEM_SIZE*blockIdx.x + SH_MEM_PER_THREAD*(threadIdx.x+1);
      for(int x = nextSect-SH_MEM_PER_THREAD; x < nextSect && x < sl; x++){
          if(x%k == 0){ sieve[x] = 0; }
      }//end loop
   }// end loop
}// end ExtendSieve

////////////////////////////////////////////////////////////////////////////////
//!InitSieve
//!Generates an initial char array of 1s
// @param sieve: char*: the sieve of Eratosthenes
////////////////////////////////////////////////////////////////////////////////
__global__ void InitSieve(char *sieve, int sl){
   int sect = SH_MEM_SIZE*blockIdx.x + SH_MEM_PER_THREAD*threadIdx.x;
   for(int x = sect; x < sect+SH_MEM_PER_THREAD && x < sl; x++){
      sieve[x]=1;
   }//end loop
} // end InitSieve

////////////////////////////////////////////////////////////////////////////////
//!InitPrimes
//!Generates an initial char array of 1s
// @param sieve: char*: the sieve of Eratosthenes
////////////////////////////////////////////////////////////////////////////////
__global__ void InitPrimes(int *primes, int pl){
   __syncthreads();
   int nextSect = SH_MEM_SIZE*blockIdx.x + SH_MEM_PER_THREAD*(threadIdx.x+1);
   for(int x = (nextSect-SH_MEM_PER_THREAD)/sizeof(int); x < nextSect/sizeof(int); x++){
      if(x < pl){primes[x]=0;}
   }//end loop
   __syncthreads();
}//end InitPrimes


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
//!ThreadIsMult
//!Each thread marks one multiple of the given prime as composite.
// @param p: int: the prime number with which we are working
// @param sieve: char*: the sieve of Eratosthenes
// @param sl: int: sieve length
////////////////////////////////////////////////////////////////////////////////
__global__ void ThreadIsMult(int p, char *sieve, int sl)
{
   int absThrInd = blockIdx.x*blockDim.x + threadIdx.x; // the absolute index
                                       // of the thread (including block index)
   int mult = p*(p+absThrInd); // the first multiple we mark is p^2
   if(mult < sl){
      sieve[mult] = 0;
   } // end if
//   __syncthreads();
} // end ThreadIsMult

////////////////////////////////////////////////////////////////////////////////
//!ThreadIsMult1
//!Each thread marks a few multiples of the given prime as composite.
//!Eliminates the need to create enough blocks for each multiple to have
//!it's own thread.
// @param p: int: the prime number with which we are working
// @param sieve: char*: the sieve of Eratosthenes
// @param sl: int: sieve length
////////////////////////////////////////////////////////////////////////////////
__global__ void ThreadIsMult1(int p, char *sieve, int sl)
{
   int absThrInd = blockIdx.x*blockDim.x + threadIdx.x; // the absolute index
                                       // of the thread (including block index)
   int absDim = blockDim.x * gridDim.x; // the total number of threads
   int mult = p*(p+absThrInd); // the first multiple we mark is p^2
   while(mult < sl){
      sieve[mult] = 0;
      mult += p*absDim;
   } // end if
//   __syncthreads();
} // end ThreadIsMult


////////////////////////////////////////////////////////////////////////////////
//!ThreadIsMult2
//!Each thread preforms removal over a range of known primes that exist
//!within the array between the current prime (p) and its square (p*p)
// @param sieve: char*: the sieve of Eratosthenes
// @param sl: int: sieve length
////////////////////////////////////////////////////////////////////////////////
__global__ void ThreadIsMult2(int *p, int pl, char *sieve, int sl)
{
   int absThrInd = blockIdx.x*blockDim.x + threadIdx.x; // the absolute index
                                       // of the thread (including block index)
   int absDim = blockDim.x * gridDim.x; // the total number of threads
   
   __syncthreads();
   
   int i = 0; 
   while(i < pl){// remove all primes within range of p to (p*p)-1
      int k = p[i];
      int mult = k*(k+absThrInd); // the first multiple we mark is p^2
      while(mult < sl){
         sieve[mult] = 0;
         mult += k*absDim;
      } // end if
      i++;
      __syncthreads();
    }//end loop
    __syncthreads();
} // end ThreadIsMult
#endif // #ifndef _ERATOSTHESES_KERNEL_H_
