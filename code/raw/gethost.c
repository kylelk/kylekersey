#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int main(int argc, char *argv[ ]) {
    struct hostent *h;
    
    /* error check the command line */
    if(argc != 2) {
        fprintf(stderr, "Usage: %s hostname\n", argv[0]);
        exit(1);
    }
    
    /* get the host info */
    if((h=gethostbyname(argv[1])) == NULL) {
        herror("gethostbyname(): ");
        exit(1);
    }
    else
        
        printf("Hostname: %s\n", h->h_name);
        printf("IP Address: %s\n", inet_ntoa(*((struct in_addr *)h->h_addr)));
        char ip[128];
        
    
    return 0;
}