#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main()
{
    int i=0;
    char *s = "9230ee4bb08660415241408b1d5348af9a049132f45d54a315e57c6a6e73aea5";
    int length = strlen(s);
    double len_root = sqrt(length);
    int line_size = ceil(len_root);
    
    while (i<length) {
        int n;
        for (n=i; n<i+line_size; n++) {
            printf("%c", s[n]);
        }
        printf("\n");
        i = i+line_size;
    }
}