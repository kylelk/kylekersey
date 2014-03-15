#include <stdio.h>
#include <string.h>

int main()
{
    char *string;
    char input[80];
    int x;
    
    x = 0;
    printf("Enter your string\n");
    fgets(input, sizeof(input), stdin);
    string = input;
    
    for (int i=0; i<strlen(string); i++) {
        for (int a=0; <#condition#>; <#increment#>) {
            <#statements#>
        }
    }
    }
    printf("%d\n", x);
    return 0;
} 