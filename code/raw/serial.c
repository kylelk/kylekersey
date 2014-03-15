#include <sys/un.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syslog.h>
#include <termios.h>

#define SERIALPORT_IS_CONSOLE

main()
{
    struct termios  tty;
    struct termios  savetty;
    speed_t     spd;
    unsigned int    sfd;
    unsigned char   buf[80];
    int     reqlen = 79;
    int     rc;
    int     rdlen;
    int     pau = 0;

#ifdef SERIALPORT_IS_CONSOLE
    sfd = STDIN_FILENO;
#else
    sfd = open("/dev/tty.usbmodemfa221", O_RDWR | O_NOCTTY | O_NONBLOCK);
#endif
    if (sfd < 0) {
        syslog(LOG_DEBUG, "failed to open: %d, %s", sfd, strerror(errno));
        exit (-1);
    }
    syslog(LOG_DEBUG, "opened sfd=%d for reading", sfd);

    rc = tcgetattr(sfd, &tty);
    if (rc < 0) {
        syslog(LOG_DEBUG, "failed to get attr: %d, %s", rc, strerror(errno));
        exit (-2);
    }
    savetty = tty;    /* preserve original settings for restoration */

    spd = B115200;
    cfsetospeed(&tty, (speed_t)spd);
    cfsetispeed(&tty, (speed_t)spd);

    cfmakeraw(&tty);

    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 10;

    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;    /* no HW flow control? */
    tty.c_cflag |= CLOCAL | CREAD;
    rc = tcsetattr(sfd, TCSANOW, &tty);
    if (rc < 0) {
        syslog(LOG_DEBUG, "failed to set attr: %d, %s", rc, strerror(errno));
        exit (-3);
    }

    do {
        unsigned char   *p = buf;

        rdlen = read(sfd, buf, reqlen);
        if (rdlen > 0) {
            if (*p == '\r')
                pau = 1;
            syslog(LOG_DEBUG, "read: %d, 0x%x 0x%x 0x%x", \
                     rdlen, *p, *(p + 1), *(p + 2));
        } else {
            syslog(LOG_DEBUG, "failed to read: %d, %s", rdlen, strerror(errno));
        }
    } while (!pau);

    tcsetattr(sfd, TCSANOW, &savetty);
    close(sfd);
    exit (0);
}