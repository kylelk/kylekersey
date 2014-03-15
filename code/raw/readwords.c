
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_LENGTH 256
#define MAX_WORDS_NUM 30000

int read_words(FILE *file, char **words, int max_size);
void display(char **words, int count);
void free_memory(char **words, int count);


void main(int argc, char **argv)
{
	char *words[MAX_WORDS_NUM];
	int count;
	FILE *file;

	if (argc > 1) {
		file = fopen(argv[1], "r");
		if (file == 0) {
			printf("Cann't open file %s!\n", argv[1]);
			exit(1);
		}
	} else {
		file = stdin;
	}


	count = read_words(file, words, MAX_WORDS_NUM);
	display(words, count);
	free_memory(words, count);
}


/**
 *	read line by line from standard input, break each line into words,
 *  and store those words into a string array.
 * 
 * preconditions: a (char *) array of size MAX_WORDS_NUM.
 * postcondistions: elements of the string array has been allocated memory and stored words
 * return: a number of words have been parsed.
 */
int read_words(FILE *file, char **words, int max_size)
{
	char line[LINE_LENGTH];
	char *token;
	int index = 0;

	while (fgets(line, LINE_LENGTH, file) != 0 && index < max_size) {
		token = strtok(line, " \t\n,;.()");
		if (token != 0) {
			words[index] = malloc(strlen(token)+1);
			strcpy(words[index], token);
			index ++;
			while ((token = strtok(0, " \t\n,;.()")) != 0 && index < max_size) {
				words[index] = malloc(strlen(token)+1);
				strcpy(words[index], token);
				index ++;
			}
		}
	}

	return index;
}

/**
 *  display each word in the string array in one line.
 *
 * preconditions: a string array which stored words with the number of words.
 */
void display(char **words, int count)
{
	int i;

	for (i = 0; i < count; i ++) {
		printf("%s\n", words[i]);
	}
}

/**
 * free memory has been allocated in the string array.
 *
 * preconditions: a string array whose elements have been allocted memory with the number of elements
 * postconditions: the memory used the elements has been released.
 *
 */
void free_memory(char **words, int count)
{
	int i;

	for (i = 0; i < count; i ++) {
		free(words[i]);
	}

}


