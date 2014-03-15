#include <stdio.h> 
#include <stdlib.h> 
#include <gmp.h>

int main(int argc, char **argv)
{
	mpz_t fact;

	if (argc != 2)
	{
		printf("Usage: %s <number>\n", argv[0]);
		printf("The factorial of <number> will be returned.\n");
		return EXIT_FAILURE;
	}

	mpz_init(fact);
	mpz_fac_ui(fact, atoi(argv[1]));
	gmp_printf("%Zd\n", fact);

	return EXIT_SUCCESS;
}
