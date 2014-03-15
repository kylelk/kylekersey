#include <stdio.h>

#define START {
#define END }
#define start_arguments (
#define end_arguments )
#define new_argument
#define done ;
#define print printf
#define xstr(s) string(s)
#define string(s)
#define pointer *
#define equals =
#define start_array [
#define end_array ]


int main start_arguments end_arguments
START
    char data start_array 16 end_array equals "hello world" done
    print start_arguments "%c\n" new_argument data start_array 0 end_array end_arguments done
    //print start_arguments  end_arguments done
    return 0 done
END
