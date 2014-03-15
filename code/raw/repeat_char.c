#include <stdio.h>

void printchar(char c[128], int n)
{
    int i;
    for(i=0;i<n;i++)
    {
        printf("%s", c);
    }
    printf("\n");
}

int main()
{
    printchar("*", 10);
}


