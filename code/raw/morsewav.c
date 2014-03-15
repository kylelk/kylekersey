//
// morsewav -- Convert text to morse code to audio WAV file.
//
// The goal was to create a morse code ring tone for an iPhone.
// - generate WAV file using this tool
// - import WAV into iTunes; create AAC version (M4A)
// - locate M4A file and rename to M4R
// - import M4R into iTunes (will now be in RingTone folder)
// - sync iPhone; select Settings -> Sounds -> Ringtone
//
// 14-Sep-2010 tvb www.LeapSecond.com/tools
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int Debug;      // debug mode
int Play;       // play WAV file
int Show;       // Show morse code

char *Path;     // output filename
double Tone;    // tone frequency (Hz)
#define EPW 50  // elements per word (definition)
double Wpm;     // words per minute
double Eps;     // elements per second (frequency of basic morse element)
double Bit;     // duration of basic morse element,cell,quantum (seconds)
double Sps;     // samples per second (WAV file, sound card)

short *pcm_data;
long pcm_count;

void wav_write (char *path, short *data, long count);
long wav_size;

// Generate one quantum of silence or tone in PCM/WAV array.

void tone (int on_off)
{
    double ampl = 0.85 * 32767;
    double pi = 3.1415926535;
    double w = 2.0 * pi * Tone;
    long i, n, size;
    static long seconds;

    if (pcm_data == NULL) {
        seconds = 1;
        size = (long) (Sps * sizeof pcm_data[0] * seconds);
        pcm_data = malloc(size);
    }

    n = (long) (Bit * Sps);
    for (i = 0; i < n; i += 1) {
        double t = (double)i / Sps;
        if (pcm_count == Sps * seconds) {
            seconds += 1;
            size = (long) (Sps * sizeof pcm_data[0] * seconds);
            pcm_data = realloc(pcm_data, size);
        }
        pcm_data[pcm_count++] = (short) (on_off * ampl * sin(w * t));
    }

    Debug && printf(on_off ? "O" : "-");
}

// Morse code table(s).

char *morse_az[] = {
    ".-",       // a
    "-...",     // b
    "-.-.",     // c
    "-..",      // d
    ".",        // e
    "..-.",     // f
    "--.",      // g
    "....",     // h
    "..",       // i
    ".---",     // j
    "-.-",      // k
    ".-..",     // l
    "--",       // m
    "-.",       // n
    "---",      // o
    ".--.",     // p
    "--.-",     // q
    ".-.",      // r
    "...",      // s
    "-",        // t
    "..-",      // u
    "...-",     // v
    ".--",      // w
    "-..-",     // x
    "-.--",     // y
    "--.."      // z
};

char *morse_09[] = {
    "-----",    // 0
    ".----",    // 1
    "..---",    // 2
    "...--",    // 3
    "....-",    // 4
    ".....",    // 5
    "-....",    // 6
    "--...",    // 7
    "---..",    // 8
    "----."     // 9
};

//
// Define dit, dah, end of letter, end of word.
//
// The rules of 1/3 and 1/2/4:
//   Morse code is: tone for one unit (dit) or three units (dah)
//   followed by the sum of one unit of silence (always),
//   plus two units of silence (if end of letter),
//   plus four units of silence (if also end of word).
//

void dit() { tone(1); tone(0); }
void dah() { tone(1); tone(1); tone(1); tone(0); }
void eol() { tone(0); tone(0); }
void eow() { tone(0); tone(0); tone(0); tone(0); }

// Generate one letter (symbol).

void morse_letter (char letter)
{
    char c, *code = "";

    if (letter >= 'a' && letter <= 'z') code = morse_az[letter - 'a'];
    if (letter >= 'A' && letter <= 'Z') code = morse_az[letter - 'A'];
    if (letter >= '0' && letter <= '9') code = morse_09[letter - '0'];

    while ((c = *code++) != '\0') {
        Show && printf("%c", c);
        if (c == '.') dit();
        if (c == '-') dah();
    }
    Show && printf(" ");
    eol();
}

// Generate one word.

void morse_word (char *word)
{
    char c;

    while ((c = *word++) != '\0') {
        morse_letter(c);
    }
    Show && printf(" ");
    eow();
}

// Display parameters.

void show_details ()
{
    double wps, ms;

    wps = Wpm / 60.0;   // words per second
    Eps = EPW * wps;    // elements per second
    ms = 1000.0 / Eps;  // milliseconds per element

    printf("%12.6lf Wpm (words per minute)\n", Wpm);
    printf("%12.6lf wps (words per second)\n", wps);
    printf("%12.6lf EPW (elements per word)\n", (double)EPW);
    printf("%12.6lf Eps (elements per second)\n", Eps);

    printf("\n");
    printf("%12.3lf ms dit\n", ms);
    printf("%12.3lf ms dah\n", ms * 3);
    printf("%12.3lf ms gap (element)\n", ms);
    printf("%12.3lf ms gap (character)\n", ms * 3);
    printf("%12.3lf ms gap (word)\n", ms * 7);

    printf("\n");
    printf("%12.3lf Hz pcm frequency\n", Sps);
    printf("%12.3lf Hz tone frequency\n", Tone);
    printf("%12.3lf    pcm/tone ratio\n", Sps / Tone);

    printf("\n");
    printf("%12.3lf Hz pcm frequency\n", Sps);
    printf("%12.3lf Hz element frequency\n", Eps);
    printf("%12.3lf    pcm/element ratio\n", Sps / Eps);

    printf("\n");
    printf("%12.3lf Hz tone frequency\n", Tone);
    printf("%12.3lf Hz element frequency\n", Eps);
    printf("%12.3lf    tone/element ratio\n", Tone / Eps);

    printf("\n");
}

// Check for sub-optimal combination of rates (poor sounding sinewaves).

int ratio_poor (double a, double b)
{
    double ab = a / b;
    long ratio = (long)(ab + 1e-6);
    return fabs(ab - ratio) > 1e-4;
}

void check_ratios (void)
{
    char nb[] = "WARNING: sub-optimal sound ratio";

    if (ratio_poor(Sps, Tone)) {
        printf("%s Sps(%lg) / Tone(%lg) = %.6lf\n", nb, Sps, Tone, Sps / Tone);
    }
    if (ratio_poor(Sps, Eps)) {
        printf("%s Sps(%lg) / Eps(%lg) = %.6lf\n", nb, Sps, Eps, Sps / Eps);
    }
    if (ratio_poor(Tone, Eps)) {
        printf("%s Tone(%lg) / Eps(%lg) = %.6lf\n", nb, Tone, Eps, Tone / Eps);
    }
}

// Define usage message.

char Usage[] =
    "Usage: morsewav [options] text...\n"
    "\n"
    "Options:\n"
    "    /wpm:#     - change words per minute (default 20)\n"
    "    /tone:#    - change tone frequency (default 900 Hz)\n"
    "    /sps:#     - change samples per second (default 44100 Hz)\n"
    "    /out:f     - change output file name (default morse.wav)\n"
    "    /play      - play audio file too\n"
    "    /debug     - debug (show timing calculations)\n"
    "\n"
    "Examples:\n"
    "    morsewav paris\n"
    "    morsewav /play cq cq cq\n"
    "    morsewav /wpm:18 /tone:1050 sos\n"
    "    morsewav /play /wpm:30 cq cq cq\n"
    ;

int get_options (int argc, char *argv[])
{
    int args = 0;

    // Set defaults.

    Wpm = 20.0;
    Tone = 900;
    Sps = 44100;
    Path = "morse.wav";
    Show = 1;

    // Get user options.

    while (argc > 1 && argv[1][0] == '/') {

        if (strcmp(argv[1], "/debug") == 0) {
            Debug = 1;
            Show = 0;
        } else

        if (strncmp(argv[1], "/tone:", 6) == 0) {
            Tone = atof(&argv[1][6]);
        } else

        if (strncmp(argv[1], "/out:", 5) == 0) {
            Path = &argv[1][5];
        } else

        if (strcmp(argv[1], "/play") == 0) {
            Play = 1;
        } else

        if (strncmp(argv[1], "/sps:", 5) == 0) {
            Sps = atof(&argv[1][5]);
        } else

        if (strncmp(argv[1], "/wpm:", 5) == 0) {
            Wpm = atof(&argv[1][5]);
        } else

        {
            fprintf(stderr, "Unknown option: %s\n", argv[1]);
            exit(1);
        }
        argc -= 1;
        argv += 1;
        args += 1;
    }

    // Note 60 seconds = 1 minute and 50 elements = 1 morse word.

    Eps = Wpm / 1.2;    // elements per second (frequency of morse coding)
    Bit = 1.2 / Wpm;    // seconds per element (period of morse coding)

    return args;
}

// Here is the main program.

void main (int argc, char *argv[])
{
    int n;

    // (1) Check options.

    if (argc < 2) {
        fprintf(stderr, Usage);
        exit(1);
    }
    n = get_options(argc, argv);
    argc -= n;
    argv += n;

    if (Debug) {
        show_details();
    }

    // (2) Generate morse code.

    printf("wave: %9.3lf Hz (/sps:%lg)\n", Sps, Sps);
    printf("tone: %9.3lf Hz (/tone:%lg)\n", Tone, Tone);
    printf("code: %9.3lf Hz (/wpm:%lg)\n", Eps, Wpm);
    check_ratios();

    while (argc > 1) {
        morse_word(argv[1]);
        argc -= 1;
        argv += 1;
    }
    printf("\n");

    // (3) Create WAV file.

    wav_write(Path, pcm_data, pcm_count);

    printf("%ld PCM samples", pcm_count);
    printf(" (%.1lf s @ %.1lf kHz)", (double)pcm_count / Sps, Sps / 1e3);
    printf(" written to %s (%.1f kB)\n", Path, wav_size / 1024.0);

    // (4) Play audio.

    if (Play) {
        char cmd[1000];
        sprintf(cmd, "mplay32.exe /play /close %s", Path);
        printf("** %s\n", cmd);
        system(cmd);
    }
}

// Create WAV file from PCM array.

typedef unsigned short WORD;
typedef unsigned long DWORD;

typedef struct _wave {
    WORD  wFormatTag;      // format type
    WORD  nChannels;       // number of channels (i.e. mono, stereo...)
    DWORD nSamplesPerSec;  // sample rate
    DWORD nAvgBytesPerSec; // for buffer estimation
    WORD  nBlockAlign;     // block size of data
    WORD  wBitsPerSample;  // number of bits per sample of mono data
} WAVE;

#define FWRITE(buf,size) \
    wav_size += size; \
    if (fwrite(buf, size, 1, file) != 1) { \
        fprintf(stderr, "Write failed: %s\n", path); \
        exit(1); \
    }

void wav_write (char *path, short *data, long count)
{
    long data_size, wave_size, riff_size;
    FILE *file;
    WAVE wave;

    if ((file = fopen(path, "wb")) == NULL) {
        fprintf(stderr, "Open failed: %s\n", path);
        exit(1);
    }

    wave.wFormatTag      = 0x1;
    wave.nChannels       = 1;
    wave.wBitsPerSample  = sizeof data[0] * 8;
    wave.nBlockAlign     = sizeof data[0] * wave.nChannels;
    wave.nSamplesPerSec  = (long)Sps;
    wave.nAvgBytesPerSec = (long)Sps * wave.nBlockAlign;

    wave_size = sizeof wave;
    data_size = count * wave.nChannels * (wave.wBitsPerSample / 8);
    riff_size = 20 + wave_size + data_size;
    FWRITE("RIFF", 4);
    FWRITE(&riff_size, 4);

    FWRITE("WAVE", 4);
    FWRITE("fmt ", 4);
    FWRITE(&wave_size, 4);
    FWRITE(&wave, wave_size);

    FWRITE("data", 4);
    FWRITE(&data_size, 4);
    FWRITE(data, data_size);

    fclose(file);
}
