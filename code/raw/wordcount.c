/*
 * wordcount.c
 *
 * A simple program to count the number of words, characters, and lines in one
 * or more text files specified on the command line. This particular
 * implementation uses two passes through the file, one to count words and the
 * other characters and lines.
 *
 * To build:
 * 		gcc -ansi -pedantic -o wordcount wordcount.c
 *
 * To run:
 * 		./wordcount <name of files to check>
 *
 * Written by Paul Bonamy - 24 May 2011
 * Updated to show more variety - 25 October 2011
 */

#include <stdlib.h> /* exit lives here */
#include <stdio.h>
#include <string.h>	/* strncmp and strlen live here */

int main(int argc, char *argv[]) {
	/*
	 * Remember: argc will end up populated with the number of arguments seen
	 * (including the name of the executable) and argv will have the actual
	 * arguments as strings. argv[argv] will always have the value 0.
	 */
	
	FILE* fp;	/* a file pointer on which to work. */
	
	char word[81]; 	/* can store up to 80 chars with room for a null byte */
	int count;		/* number of words in input */
	int charCount;	/* number of characters in input */
	int lineCount;	/* number of lines in input */
	int lineLength;	/* number of characters on a single line */
	int i = 1;
	
	/*
	 * check to make sure we actually got a reasonable number of arguments.
	 * argc is always at least 1, but we need at least 2 to give us the name
	 * of the program and at least one file to count.
	 */
	if(argc < 2) {
		/* stderr is the standard error output,
		   and is a good place to send, well, errors */
		fprintf(stderr, "No file specified. Exiting...\n");
		exit(1); /* exit kills the program with the given return value */
	}
	
	/*
	 * this loop will run us through all of the files specified at the command
	 * line. note that i starts at 1 because argv[0] will be the name of the
	 * program, which is not a file we want to try and read in text mode
	 *
	 * We can use a while loop here since argv[argc] is guaranteed to be 0
	 */
	while (argv[i]) {
		/*
		 * zero out our counter variables. C allows chaining like this
		 * because the result of an assignment is effectively returned
		 * so it can be used for other things.
		 */
		count = charCount = lineCount = 0;
		
		/*
		 * this is the first pass. We'll use fscanf to read in a word at
		 * a time on this pass.
		 */
		
		/*
		 * fopen takes the name of a file (as a string) and a mode (read, in
		 * this case) and attempts to open the given file in the given mode.
		 *
		 * it will return either a pointer to a file stream, or NULL in the
		 * event of a failure of some sort.
		 *
		 * ALWAYS check the return value before trying to use the file pointer!
		 */
		fp = fopen(argv[i], "r");
		
		/* check to see if the file opened correctly. */
		if(fp) {
		
			/* loop until we hit the end of the file */
		
			while(!feof(fp)) {
				
				/*
				 * EOF only gets set once you try to read past the end,
				 * so we can get into trouble if the file ends in whitespace.
				 *
				 * However, (f)scanf will return EOF if it hit the end of the
				 * file during a read, so we make sure that wasn't set before
				 * incrementing the count.
				 *
				 * Note: this will give inaccurate results if a single word
				 * exceeds 80 characters.
				 */
				if(fscanf(fp, "%80s", word) != EOF) {
					count++;	/* got a word, so increment count */
				}
			}
		
		}
		/* close the file when you're done with it. */
		fclose(fp);
		
		
		/*
		 * pass two: do line and character count. Note: there's no need
		 * to have two passes, but the single-pass version is left as an
		 * exercise to the reader.
		 *
		 * Note: This bit was added after class
		 */
		fp = fopen(argv[i], "r");
		if(fp) {
		
			do {
				/*
				 * this time around, we'll use fgets to read a line at a time.
				 * Note that fgets will read until it encounters an EOL/EOF, or
				 * reads 80 characters, whichever comes first.
				 */
				fgets(word, 81, fp);
				
				/*
				 * figure out the number of characters in the line and
				 * update the character count
				 */
				lineLength = strlen(word);
				charCount += lineLength;
				
				/*
				 * Note that, if a line is especially long, fgets won't
				 * read it all in one go. We have to deal with that if we
				 * want to get an accurate count.
				 */
				if(lineLength < 80) {
					/*
					 * if there are less than 80 chars, total, we know
					 * fgets hit a newline (or EOF), so update the line count.
					 */
					lineCount++;
				}
				else{
					/*
					 * there's a special case where a line is exactly 80
					 * characters including a newline. If that's the case,
					 * increment the line count.
					 */
					if(word[79] == '\n'){
						lineCount++;
					}
				}
			} while (!feof(fp));
		}
		fclose(fp);
		
		/*
		 * Note: it's entirely possible there are edge causes that this
		 * code a) doesn't catch and b) doesn't know it doesn't catch
		 *
		 * Finding them are up to the reader.
		 */
		
		/* print a little report */
		printf("%s:\t%d\t%d\t%d\n", argv[i], count, charCount, lineCount);
		
		i++;
	}

	/* done with all files, so we're done. return. */	
	return 0;
}