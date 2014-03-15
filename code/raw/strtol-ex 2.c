#include <stdio.h>
#include <stdlib.h>

int main()
{
    //3073168210851302897
    // fingerprint = 0x2aa6163496d08df1
    char String[32];
    printf("fingerprint: ");
    fgets(String, 32, stdin);
    
    int Base=16;			/* Base 16		*/
    long int Ans;			/* Result		*/
    Ans = strtol(String, NULL, Base);
    printf("%ld\n", Ans);
}