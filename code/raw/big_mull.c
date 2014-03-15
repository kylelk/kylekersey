#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

int main(void)
{
    mpz_t x;
    mpz_t y;
    mpz_t result;
    
    mpz_init(x);
    mpz_init(y);
    mpz_init(result);
    
    mpz_set_str(x, "2", 10);
    mpz_set_str(y, "9263591128439081", 10);
    
    // mpz_mul(result, x, y);
    mpz_ui_pow_ui(result, 2, 1024);
    gmp_printf("\n    %Zd\n*\n    %Zd\n--------------------\n%Zd\n\n", x, y, result);
    
    /* free used memory */
    mpz_clear(x);
    mpz_clear(y);
    mpz_clear(result);
    return EXIT_SUCCESS;
}
