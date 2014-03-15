#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>


void factorial(unsigned long long n, mpz_t result) {
    mpz_set_ui(result, 1);

    while (n > 1) {
        mpz_mul_ui(result, result, n);
        n = n-1;
    }
}

int main(int argc, char *argv[]) {
    mpz_t fact;
    mpz_init(fact);
    
    
    mpz_t maxSeach;
    mpz_init(maxSeach);

    mpz_t numberToFind;
    mpz_init_set_str(numberToFind, argv[1], 16);
    
    
    factorial(100, fact);
    mpq_div(sum, sum, count);
    
    char *as_str = mpz_get_str(NULL, 16, fact);
    printf("%s\n", as_str);

    mpz_clear(fact);
    mpz_clear(maxSeach);
    free(as_str);
}