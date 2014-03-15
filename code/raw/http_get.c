//
//  http_get.c
//  
//
//  Created by Kyle on 9/12/13.
//
//

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdbool.h>
#include <unistd.h>

#define BUFFSIZE 512

void error(char *msg);

int serverSocket;
int i = 0;
size_t bWritten, bTotal;
struct sockaddr_in serverAddr;
struct hostent *hostptr;
char *ip = "192.168.2.2";
char *tok, *host, *request, buffer[BUFFSIZE];


int main (int argc, const char * argv[]) {
    printf("%s rev.0\n", argv[0]);
    if (argc < 2)
    {
        printf("Usage: %s [address]\n", argv[0]);
        exit(-1);
    }
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(80);
    serverAddr.sin_addr.s_addr = inet_addr(ip);
    tok = strpbrk(argv[1], "/");
    host = malloc(sizeof(char) * (tok - &(*argv[1]) + 1));
    strncpy(host, argv[1], tok - &(*argv[1]) );
    //disabled because xcode has a problem with gethostbyname()
    char *test = "acenusm.com";
    hostptr = gethostbyname(test);
    printf("Resolved: %s\n", host);
    printf("%s \n", hostptr->h_addr_list[0]);
    //these are really nasty but they work.
    printf("Requesting: %s\n", argv[1] + (tok - argv[1]));
    request = malloc( (tok - argv[1]) - strlen(argv[1]) + sizeof(char) * 26 + strlen(host) + 1 );
    sprintf(request, "GET %s HTTP/1.0\r\nHOST:%s \r\n", argv[1] + (tok - argv[1]), host);
    //free and Null pointer
    if(host)
    {
        free(host);
        host = NULL;
    }
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0)
        error("socket()");
    if (connect(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
        error("connect()");
    for (bTotal = 0; bTotal < strlen(request); bTotal += bWritten)
    {
        bWritten = write(serverSocket, &request[bTotal], strlen(request) - bTotal);
        if (bWritten == 0)
            break;
    }
    //i = send(serverSocket, &request, (strlen(request) + 1), 0);
    //if (strlen(request) + 1 != i)
    //  error("send()");
    printf("waiting for response...\n");
    //assuming recv return 0 when socket is closed)
    FILE *fp;
    fp = fopen("default.txt", "a");
    for(int j; j < BUFFSIZE; j++)
    {
        i = recv(serverSocket, &buffer+j, 4, 0);
        printf("%s\n", buffer);
    }
    free(request);
    request = NULL;
    
}

void error(char *msg)
{
    printf("Error: %s\n", msg);
    perror("");
    exit(0);
}