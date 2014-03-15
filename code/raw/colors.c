#include <stdio.h>
#include <ncurses.h>
#include <curses.h>

int main(void) {
    initscr();
    start_color();
    color_set(COLOR_BLACK, COLOR_BLACK);
//    
//    init_pair(1, COLOR_BLACK, COLOR_RED);
//    init_pair(2, COLOR_BLACK, COLOR_GREEN);
//    init_pair(3, COLOR_BLACK, COLOR_BLUE);
//    
//    attron(COLOR_PAIR(1));
//    printw("This should be printed in black with a red background!\n");
//    
//    attron(COLOR_PAIR(2));
//    printw("And this in a green background!\n");
//    
//    attron(COLOR_PAIR(3));
//    printw("this Text shouls be blue");
    refresh();
    
    //getch();
    
    //endwin();
    return 0;
}