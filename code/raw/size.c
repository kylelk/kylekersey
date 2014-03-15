#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>

/* This routine returns the size of the file it is called with. */

static unsigned
get_file_size (const char * file_name)
{
    struct stat sb;
    if (stat (file_name, & sb) != 0) {
        fprintf (stderr, "'stat' failed for '%s': %s.\n",
                 file_name, strerror (errno));
        exit (EXIT_FAILURE);
    }
    return sb.st_size;
}

int main (int argc, char ** argv)
{
    int i;
    for (i = 0; i < argc; i++) {
        const char * file_name;
        unsigned size;

        file_name = argv[i];
        size = get_file_size (file_name);
        printf ("%20s has %d bytes.\n", file_name, size);
    }
    return 0;
}