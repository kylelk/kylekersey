//
//  primes2.c
//  
//
//  Created by Kyle on 8/30/13.
//
//

#include <stdio.h>
#include <math.h>
#define x 10

int main()
{
    int n;
    int p;
    int s;
    double a;
    
	for (n=1; n<x; n++) {
        p = pow(10, n);
        a = p / log(p);
        s = (x-n)+1;
        printf("---------------------------------\n");
        printf("| %d%*s| %f\n", p, s," ", a);
    }
}
