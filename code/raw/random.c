#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
    int byte_count = 64;
    char data[64];
    FILE *fp;
    fp = fopen("/Users/kyle/app-tests/c/random.c", "r"); //"/dev/urandom"
    fread(&data, 1, byte_count, fp);
    int n;
    for (n=0; n<byte_count; n++) {
        printf("%p\n", data[n]);
    }
    fclose(fp);
    printf("\n");
}