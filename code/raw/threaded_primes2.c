#include <pthread.h>
#include <stdio.h>

void* compute_prime (void* arg)
{
  int candidate = 2;
  printf("Threads: %d\n", candidate);
  int n = *((int*) arg);

  while (1) {
    int factor;
    int is_prime = 1;
    
    for (factor = 2; factor < candidate; ++factor)
      if (candidate % factor == 0) {
        is_prime = 0;
        break;
      }
    
    if (is_prime) {
      if (--n == 0)
        return (void*) candidate;
    }
    ++candidate;
  }
  return NULL;
}

int main ()
{
  pthread_t thread;
  int which_prime = 5000;
  int prime;
  printf("\n%d\n", which_prime);

  pthread_create (&thread, NULL, &compute_prime, &which_prime);
  pthread_join (thread, (void*) &prime);
  
  printf("%d\n", prime);
  return 0;
}