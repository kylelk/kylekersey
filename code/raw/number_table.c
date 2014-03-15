//
//  divisors.c
//  
//
//  Created by Kyle on 8/23/13.
//
//

#include <stdio.h>
#include <math.h>
//#define N 10

int main()
{
    int n;
    int l;
    int input;
    int N;
    
    printf("enter the N\n");
    scanf("%d", &input);
    
    N = input;
    for (n=1; n<=N; n++) { //generate a list of numbers
        //printf("%d ", n);
        printf("\n");
        for (l=n; l<=N; l++) {
            printf("%d ", l);
        }
    }
    printf("\n");
}

