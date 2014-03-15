////////////////////////////////////////////////////////////////////////////////
//
// Guo, Jackson, Jiang, and Pape
// Professor Annexstein
// Parallel Computing
// 19 April 2011
// Lab 2: Sieve of Eratosthenes
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _ERATOSTHENES_H_
#define _ERATOSTHENES_H_

// the size of shared memory in bytes
#define SH_MEM_SIZE /*(12383) /*/ (16384) /**/// somehow, we have been using too much memory
// each thread's portion of shared memory in bytes, assuming 512 threads
#define SH_MEM_PER_THREAD (32)
// number of ints that fit in shared memory
#define SH_MEM_INT (SH_MEM_SIZE/sizeof(int))


////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"


////////////////////////////////////////////////////////////////////////////////
//! eratosLinear
//! linear implementation of the sieve of eratosthenes
//  @param num: int indicating the number of numbers to search for primes
//  @param sieve: char array of size num initialized to 1's (or NULL)
//  @return: the number of primes between 0 and 2^pow - 1
////////////////////////////////////////////////////////////////////////////////
int eratosLinear(int num, char * sieve);

////////////////////////////////////////////////////////////////////////////////
//! findsmallest
//! finds the smallest prime above min
//  @param sieve: array of unsigned characters indicating whether or not a
//    factor has been found for a given index
//  @param num: int indicating the size of primes
//  @param min: int indicating the index at which to begin searching
//  @return: int indicating the index at which a prime has been found 
////////////////////////////////////////////////////////////////////////////////
int findsmallest(const char * const sieve, int num, int min);

////////////////////////////////////////////////////////////////////////////////
//! sumLinear
//! computes the sum of the numbers passed
//  @param sieve: array of unsigned characters indicating whether or not a
//    each index is prime
//  @param num: int indicating the size of primes
//  @return: int, the number of primes (sum of the values in primes)
////////////////////////////////////////////////////////////////////////////////
int sumLinear(const char * const sieve, int num);

#endif // _ERATOSTHENES_H_

