//
//  window.c
//  
//
//  Created by Kyle on 9/14/13.
//
//

#include <stdlib.h>
#include <stdio.h>
#include <ncurses.h>

#define WORLD_WIDTH 50
#define WORLD_HEIGHT 20

int main(int argc, char *argv[]) {
    
    WINDOW *snakeys_world;
    int offsetx, offsety;
    
    initscr();
    refresh();
    
    offsetx = (COLS - WORLD_WIDTH) / 2;
    offsety = (LINES - WORLD_HEIGHT) / 2;
    
    snakeys_world = newwin(WORLD_HEIGHT,
                           WORLD_WIDTH,
                           offsety,
                           offsetx);
    
    box(snakeys_world, 0 , 0);
    
    wrefresh(snakeys_world);
    
    while (1) {
        delwin(snakeys_world);
    }
    
//    delwin(snakeys_world);
//    
//    endwin();
    
    return 0;
    
}

