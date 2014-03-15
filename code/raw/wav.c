////#include <stdio.h>
////#include <stdlib.h>
////#include <string.h> // for memcmp
////#include <stdint.h> // for int16_t and int32_t
////
////struct wavfile
////{
////    char    id[4];          // should always contain "RIFF"
////    int32_t totallength;    // total file length minus 8
////    char    wavefmt[8];     // should be "WAVEfmt "
////    int32_t format;         // 16 for PCM format
////    int16_t pcm;            // 1 for PCM format
////    int16_t channels;       // channels
////    int32_t frequency;      // sampling frequency
////    int32_t bytes_per_second;
////    int16_t bytes_by_capture;
////    int16_t bits_per_sample;
////    char    data[4];        // should always contain "data"
////    int32_t bytes_in_data;
////} __attribute__((__packed__));
////
////int is_big_endian(void) {
////    union {
////        uint32_t i;
////        char c[4];
////    } bint = {0x01000000};
////    return bint.c[0]==1;
////}
////
////int main(int argc, char *argv[]) {
////    char *filename=argv[1];
////    FILE *wav = fopen(filename,"rb");
////    struct wavfile header;
////    
////    if ( wav == NULL ) {
////        fprintf(stderr,"Can't open input file %s\n", filename);
////        exit(1);
////    }
////    
////    // read header
////    if ( fread(&header,sizeof(header),1,wav) < 1 ) {
////        fprintf(stderr,"Can't read input file header %s\n", filename);
////        exit(1);
////    }
////    
////    // if wav file isn't the same endianness than the current environment
////    // we quit
////    if ( is_big_endian() ) {
////        if (   memcmp( header.id,"RIFX", 4) != 0 ) {
////            fprintf(stderr,"ERROR: %s is not a big endian wav file\n", filename);
////            exit(1);
////        }
////    } else {
////        if (   memcmp( header.id,"RIFF", 4) != 0 ) {
////            fprintf(stderr,"ERROR: %s is not a little endian wav file\n", filename);
////            exit(1);
////        }
////    }
////    
////    if (   memcmp( header.wavefmt, "WAVEfmt ", 8) != 0
////        || memcmp( header.data, "data", 4) != 0
////        ) {
////        fprintf(stderr,"ERROR: Not wav format\n");
////        exit(1);
////    }
////    if (header.format != 16) {
////        fprintf(stderr,"\nERROR: not 16 bit wav format.");
////        exit(1);
////    }
////    fprintf(stderr,"format: %d bits", header.format);
////    if (header.format == 16) {
////        fprintf(stderr,", PCM");
////    } else {
////        fprintf(stderr,", not PCM (%d)", header.format);
////    }
////    if (header.pcm == 1) {
////        fprintf(stderr, " uncompressed" );
////    } else {
////        fprintf(stderr, " compressed" );
////    }
////    fprintf(stderr,", channel %d", header.pcm);
////    fprintf(stderr,", freq %d", header.frequency );
////    fprintf(stderr,", %d bytes per sec", header.bytes_per_second );
////    fprintf(stderr,", %d bytes by capture", header.bytes_by_capture );
////    fprintf(stderr,", %d bits per sample", header.bytes_by_capture );
////    fprintf(stderr,"\n" );
////    
////    if ( memcmp( header.data, "data", 4) != 0 ) {
////        fprintf(stderr,"ERROR: Prrroblem?\n");
////        exit(1);
////    }
////    fprintf(stderr,"wav format\n");
////    
////    // read data
////    long long sum=0;
////    int16_t value;
////    int i=0;
////    fprintf(stderr,"---\n", value);
////    while( fread(&value,sizeof(value),1,wav) ) {
////        if (value<0) { value=-value; }
////        sum += value;
////    }
////    printf("%lld\n",sum);
////    exit(0);
////}
//#include <stdio.h>
//#include <stdlib.h>
//#include <stdint.h>
//
//struct wavfile
//{
//    char        id[4];          // should always contain "RIFF"
//    int     totallength;    // total file length minus 8
//    char        wavefmt[8];     // should be "WAVEfmt "
//    int     format;         // 16 for PCM format
//    short     pcm;            // 1 for PCM format
//    short     channels;       // channels
//    int     frequency;      // sampling frequency
//    int     bytes_per_second;
//    short     bytes_by_capture;
//    short     bits_per_sample;
//    char        data[4];        // should always contain "data"
//    int     bytes_in_data;
//};
//
//int main(int argc, char *argv[]) {
//    char *filename=argv[1];
//    FILE *wav = fopen(filename,"rb");
//    struct wavfile header;
//    
//    if ( wav == NULL ) {
//        fprintf(stderr,"Can't open input file %s\n", filename);
//        exit(1);
//    }
//    
//    // read header
//    if ( fread(&header,sizeof(header),1,wav) < 1 )
//    {
//        fprintf(stderr,"Can't read file header\n");
//        exit(1);
//    }
//    if (    header.id[0] != 'R'
//        || header.id[1] != 'I'
//        || header.id[2] != 'F'
//        || header.id[3] != 'F' ) {
//        fprintf(stderr,"ERROR: Not wav format\n");
//        exit(1);
//    }
//    
//    fprintf(stderr,"wav format\n");
//    
//    // read data
//    long sum=0;
//    short value=0;
//    while( fread(&value,sizeof(value),1,wav) ) {
//        // fprintf(stderr,"%d\n", value);
//        if (value<0) { value=-value; }
//        sum += value;
//    }
//    printf("%ld\n",sum);
//    exit(0);
//}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* M_PI is declared in math.h */
#define PI M_PI


typedef unsigned int	UI;
typedef unsigned long int	UL;
typedef unsigned short int	US;
typedef unsigned char	UC;
typedef signed int		SI;
typedef signed long int	SL;
typedef signed short int	SS;
typedef signed char	SC;


#define attr(a) __attribute__((a))

#define packed attr(packed)

/* WAV header, 44-byte total */
typedef struct{
    UL riff	packed;
    UL len	packed;
    UL wave	packed;
    UL fmt	packed;
    UL flen	packed;
    US one	packed;
    US chan	packed;
    UL hz	packed;
    UL bpsec	packed;
    US bpsmp	packed;
    US bitpsmp	packed;
    UL dat	packed;
    UL dlen	packed;
}WAVHDR;



int savefile(const char*const s,const void*const m,const int ml){
    FILE*f=fopen(s,"wb");
    int ok=0;
    if(f){
        ok=fwrite(m,1,ml,f)==ml;
        fclose(f);
    }
    return ok;
}


/* "converts" 4-char string to long int */
#define dw(a) (*(UL*)(a))


/* Makes 44-byte header for 8-bit WAV in memory
 usage: wavhdr(pointer,sampleRate,dataLength) */

void wavhdr(void*m,UL hz,UL dlen){
    WAVHDR*p=m;
    p->riff=dw("RIFF");
    p->len=dlen+44;
    p->wave=dw("WAVE");
    p->fmt=dw("fmt ");
    p->flen=0x10;
    p->one=1;
    p->chan=1;
    p->hz=hz;
    p->bpsec=hz;
    p->bpsmp=1;
    p->bitpsmp=8;
    p->dat=dw("data");
    p->dlen=dlen;
}


/* returns 8-bit sample for a sine wave */
UC sinewave(UL rate,float freq,UC amp,UL z){
    return sin(z*((PI*2/rate)*freq))*amp+128;
}


/* make arbitrary audio data here */
void makeaud(UC*p,const UL rate,UL z){
    float freq=500;
    UC amp=120;
    while(z--){
        *p++=sinewave(rate,freq,amp,z);
    }
}


/* makes wav file */
void makewav(const UL rate,const UL dlen){
    const UL mlen=dlen+44;
    UC*const m=malloc(mlen);
    if(m){
        wavhdr(m,rate,dlen);
        makeaud(m+44,rate,dlen);
        savefile("out.wav",m,mlen);
    }
}


int main(){
    if(sizeof(WAVHDR)!=44)puts("bad struct");
    makewav(22050,64000);
    return 0;
}