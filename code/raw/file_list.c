/*
 * This program displays the names of all files in the current directory.
 */

#include <dirent.h>
#include <stdio.h>
#include <string.h>

int EndsWith(const char *str, const char *suffix)
{
    if (!str || !suffix)
        return 0;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

int main(int argc, char *argv[])
{
    DIR           *d;
    struct dirent *dir;
    d = opendir(".");
    char *names;
    char *extension_name = argv[1];
    int count = 0;
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            char *file_name = dir->d_name;
            //printf("%s\n", dir->d_name);
            if (EndsWith(file_name, extension_name)==1) {
                printf("%d    %s\n",count ,file_name);
                //names[count] = file_name;
                memcpy(names, file_name, strlen(file_name));
                count+=1;
            }
        }
        
        closedir(d);
        
    }
    
    return(0);
}
