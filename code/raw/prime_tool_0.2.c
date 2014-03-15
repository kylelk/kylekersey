/*! Generate a 1024 bits prime using GMP library
 \Author switcher, juin 2007
 \code gcc -Wall -g -o prime_tool_0.2 prime_tool_0.2.c `pkg-config --cflags --libs glib-2.0` -lgmp
 */

#include <stdio.h>
#include <gmp.h>

int main ()
{
	/*! create and init a base 10 integer called "bign"
	 */
	mpz_t bign;
	mpz_init_set_str (bign, "0", 10);

	mpz_t zero;
	mpz_init_set_str (zero, "0", 10);

	/*! init random state
	 */
	gmp_randstate_t rstate;
	gmp_randinit_default (rstate);

	int rep = 0;

	while( rep  == 0)
	{

		/*! gen a 1024 bits number and store it in bign
		*/
		mpz_urandomb(bign, rstate, 1024);

		//gmp_printf("testing %Zd\n", bign);

		/*! called GMP primality test (trivial div + Rabin-Miller)
		*/
		rep = mpz_probab_prime_p(bign, 10);
	}

	gmp_printf("PRIME = %Zd\n",bign);

	/*! clear before exit
	 */
	mpz_clear (bign);

	return 0;
}
