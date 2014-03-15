//
//  table.c
//  
//
//  Created by Kyle on 9/4/13.
//
//

#include <stdio.h>

int main()
{
    double num=123412341234.123456789;
    char *output;
    sprintf(output,"%f",num);
    printf("%s",output);
}