/*
 * Purpose : to read a C source code file , scan and identify all single and
 *           multiline comments and generate an output file with, comments trimmed off.
 * Author  : Anand Kumar Mishra
 * Date    : 15th Feb 2013, 2312 HRS IST
 *
 * Important Note : beware of the following cases
 *    1. printf("<SLCS> hello world<SLCE>"); == not treating comment
 *    2. printf("<MLCS>hellow world<MLCE>"); == not treating comment
 *    3. <MLCS>message meassage.... printf("<MLCE>") <MLCE>
 *       the first <MLCE> will be considered the end of comment
 *       and will result in error actually
 *  Fault : using 256 char buffer to read; look ahead is not possible for 255th index
 *          since thats the last character, program may miss some comments in such cases.
 *
 *          solutions:
 *              1. if buffer size is larger than line size of program the risk decreases
 *              2. if a half match is done read next chunk and analyze using it.
 *
 *   SLCS : single line comment start token (//)
 *   MLCS : multiline comment start token (/*)
 *   SLCE : single line comment stop/end token (a new line character)
 *   MLCE : multiline comment stop/end token (* /) without spaces.
 *
 *
 * */

#include <stdio.h>
#include <string.h>

#define TRUE 	1
#define FALSE	0

//Comment flag values
#define NOT_A_COMMENT	   0
#define MULTLINE_COMMENT   1
#define SINGLELINE_COMMENT 2

//String Flag values
#define NOT_A_STRING 0
#define IS_STRING    1

#define __BUF_SZ 512

void usage(char *prog_name){

	fprintf(stderr,"Usage : \n"
		   "%s <input_file_name> <output_file_name>\n"
			"Note: Beware if persmissions allow and a file with name "
			"specified by <output_file_name> exsist then it will be overwritten.\n",prog_name);
}

int main(int argc, char *argv[]){

  char *input_file,*output_file;
  FILE *fin,*fout;
  //To be used for buffered reading
  char input_buffer[__BUF_SZ];
  char output_buffer[__BUF_SZ];

  //flags
  unsigned char string_flag		= NOT_A_STRING;
  unsigned char comment_flag		= NOT_A_COMMENT;
  unsigned char store_in_file_flag	= FALSE;
  unsigned int chars_read_count		= 0;
  unsigned int index			= 0;
  unsigned int output_buffer_index	= 0;

  //Verfiying the command line arguments count.
  if(argc<3){

	  //Missing some required arguments; print usage and abort.
	  usage(argv[0]);
	  return 1;
  }

  //fixing the char pointers to point to specified file name
  input_file=argv[1];
  output_file=argv[2];

  if(!(fin=fopen(input_file,"r"))){

	  fprintf(stderr,"Unable able to open input file for reading\n"
			  "please ensure that file permissions are in good state, or incase of a typo.\n");
	  return 2;
  }

  if(!(fout=fopen(output_file,"w"))){

  	  fprintf(stderr,"Unable able to open output file for writing\n"
  			  "please ensure that file permissions are in good state, or incase of a typo.\n");
  	  return 3;
  }

  //Reading the input file
  while(!feof(fin)){
    
    fgets(input_buffer,__BUF_SZ,fin);
    //printf("%s",input_buffer);

    //Correct erroneous reading, to protect program from giving SegFault
    input_buffer[__BUF_SZ-1]='\0';

    chars_read_count=strlen(input_buffer);
    //Scan for comments and dump in output file
    for(index=0,output_buffer_index=0;index<chars_read_count;index++)
    {
    	store_in_file_flag=TRUE;

    	if(input_buffer[index]=='"')
    	{
    		//identified a string token
    		if(comment_flag == NOT_A_COMMENT)
    		{
    			//check if this string is already in a string/char
    			//using look ahead, and/or look backward
    			if(index>0 && input_buffer[index-1]=='\\') // taking care of strings like printf("\"hello world\"");
    				string_flag=IS_STRING;
    			else if(index>0 && index<__BUF_SZ-1 && input_buffer[index-1]=='\'' && input_buffer[index+1]=='\'') // taking care of strings like printf("%c",'"');
    				string_flag=NOT_A_STRING;
    			else if(string_flag==IS_STRING)
    				string_flag=NOT_A_STRING;
    			else
    				string_flag=IS_STRING;
    		}
    	}else if(input_buffer[index]=='\n' ){
			//identified a  single line comment stop token
    		if(string_flag==NOT_A_STRING &&
    				(comment_flag == SINGLELINE_COMMENT))
    			comment_flag= NOT_A_COMMENT;
	}else if(index<__BUF_SZ-1 && input_buffer[index]=='/' && input_buffer[index+1]=='/'){
			//using look ahead to verify if we got single line comment start token
    		if(string_flag==NOT_A_STRING &&
    				(comment_flag!=MULTLINE_COMMENT))
    		{
    			comment_flag=SINGLELINE_COMMENT;
    			//skip the next charater since already analysed
    			index+=1;
    			//Not storing comment tokens
    			store_in_file_flag=FALSE;
    		}
	}else if(index<__BUF_SZ-1 && input_buffer[index]=='/' && input_buffer[index+1]=='*'){
			//using look ahead to verify if we got multi line comment start token
    		if(string_flag==NOT_A_STRING &&
    				(comment_flag != SINGLELINE_COMMENT || comment_flag!=MULTLINE_COMMENT))
    		{
    			comment_flag=MULTLINE_COMMENT;
    			//skip the next charater since already analysed
    			index+=1;
    			//Not storing comment tokens
    			store_in_file_flag=FALSE;
    		}
	}else if(index<__BUF_SZ-1 && input_buffer[index]=='*' && input_buffer[index+1]=='/'){
			//using look ahead to verify if we got multi line comment stop token
    		if(comment_flag == MULTLINE_COMMENT)
    		{
    			comment_flag=NOT_A_COMMENT;
    			//skip the next charater since already analysed
    			index+=1;
    			//Not storing comment tokens
    			store_in_file_flag=FALSE;
    		}
	}

    	if(store_in_file_flag==TRUE && comment_flag==NOT_A_COMMENT)
    	{
    		output_buffer[output_buffer_index++]=input_buffer[index];
    	}
    }

    //since output_buffer is of same size as of input buffer it will never be overflow
    //in one pass, hence dumping its content in file after the end of pass.
    //trucate the unsed buffer, content if any.
    output_buffer[output_buffer_index++]='\0';
    fputs(output_buffer,fout);
  }
  fclose(fin);
  fclose(fout);
}
