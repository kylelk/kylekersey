#include <stdio.h>
typedef unsigned int crypto_t;
crypto_t pwrmod( crypto_t a, crypto_t b, crypto_t n ) {
    crypto_t r;
        if ( b == 0 ) return 1;
        if ( b % 2 == 1 ) return (a*pwrmod( a, b-1, n )) % n;
        r= pwrmod( a, b/2, n );
    return (r*r) % n;
}
int main( void ) {
        printf( "%u\n", (unsigned int) pwrmod( 20232, 5, 23393 ) );
    return 0;
}
