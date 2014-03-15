//
//  sum.c
//  
//
//  Created by Kyle on 9/4/13.
//
//

#include <stdio.h>
#include <math.h>

int main()
{
    float n, count, sum=0;
    printf("Enter an integer: ");
    scanf("%f",&n);
    for(count=1;count<=n;++count)  
    {
        sum+=count;                
    }
    printf("Sum = %.2f\n", sum);
    return 0;
}
