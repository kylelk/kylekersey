#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    while (1) {
        
        time_t rawtime;
        char st[30];
        struct tm * timeinfo;
        
        time ( &rawtime );
        timeinfo = localtime ( &rawtime );
        sprintf (st,"%s", asctime (timeinfo));
        *(index(st,'\n'))='\0';
        printf("\r%s",st);
        fflush(stdout);
        sleep(1);
    }
    return 0;
}
