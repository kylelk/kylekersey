#include <stdio.h>
int main(){
    int n;
    float a=1;
    for (n=0; n<=100; n++) {
        a=1/(a+1);
    }
    printf("%f\n", a);
    return 0;
}
