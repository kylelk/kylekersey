#include <sys/types.h> 
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <string.h> 
#include <stdio.h> 
#include <stdlib.h>

int main(void)
{
    int socket_handle;
    struct sockaddr_in socket_detials;
    char *input_buffer;
    char *pinput_buffer;
    ssize_t bytes_received;
    ssize_t bytes_sent;
    char *phttpget;
    char *httpget = "GET / HTTP/1.0\r\n" "Host: www.google.com\r\n" "\r\n";

    phttpget = httpget;
    bytes_sent = 0;

    input_buffer = malloc(1024);
    if (input_buffer == NULL)
    {
        printf("Sorry, couldnt allocate memory for input buffer\n");
        return -1;
    }
    memset(input_buffer, 0, 1024);

    memset(&socket_detials, 0, sizeof(struct sockaddr_in));

    socket_handle = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_handle == -1)
    {
        printf("Could not create socket\n");
        return -1;
    }
    socket_detials.sin_family = AF_INET;
    socket_detials.sin_addr.s_addr = inet_addr("173.194.33.2");
    socket_detials.sin_port = htons(80);

    if (connect
        (socket_handle, (struct sockaddr *)&socket_detials,
         sizeof(struct sockaddr)) == -1)
    {
        printf("Couldnt connect to server\n");
        return -1;
    }

    printf("Attempting to send %zd bytes to server\n", strlen(httpget));
    for (;;)
    {
        bytes_sent = send(socket_handle, phttpget, strlen(phttpget), 0);
        if (bytes_sent == -1)
        {
            printf("An error occured sending data\n");
            return -1;
        }
        if (httpget + strlen(httpget) == phttpget)
            break;
        phttpget += bytes_sent;
    }

    for (;;)
    {
        bytes_received = recv(socket_handle, input_buffer, 1023, 0);
        if (bytes_received == -1)
        {
            printf("An error occured during the receive procedure \n");
            return 0;
        }
        if (bytes_received == 0)
            break;
        pinput_buffer = input_buffer + bytes_received;
        *pinput_buffer = 0;
        printf("%s", input_buffer);
    }

    printf("\nFinished receiving data\n");
    return 0;
}
