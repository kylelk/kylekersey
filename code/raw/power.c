#include <stdio.h> 
#include <stdlib.h> 
#include <gmp.h>

int main(int argc, char **argv)
{
	mpz_t pow;

//	if (argc != 2)
//	{
//		printf("Usage: %s <number>\n", argv[0]);
//		printf("The factorial of <number> will be returned.\n");
//		return EXIT_FAILURE;
//	}

	mpz_init(pow);
    mpz_ui_pow_ui(pow, atoi(argv[1]), atoi(argv[2]));
	//mpz_fac_ui(fact, atoi(argv[1]));
	gmp_printf("%Zd\n", pow);

	return EXIT_SUCCESS;
}
