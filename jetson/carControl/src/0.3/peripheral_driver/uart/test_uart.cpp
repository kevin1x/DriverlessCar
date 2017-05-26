
#include "api_uart.h"

#include <termios.h>
#include <iostream>

using namespace std;

int getkey()
{
    int character;
    struct termios orig_term_attr;
    struct termios new_term_attr;

    /* set the terminal to raw mode */
    tcgetattr(fileno(stdin), &orig_term_attr);
    memcpy(&new_term_attr, &orig_term_attr, sizeof(struct termios));
    new_term_attr.c_lflag &= ~(ECHO|ICANON);
    new_term_attr.c_cc[VTIME] = 0;
    new_term_attr.c_cc[VMIN] = 0;
    tcsetattr(fileno(stdin), TCSANOW, &new_term_attr);

    /* read a character from the stdin stream without blocking */
    /*   returns EOF (-1) if no character is available */
    character = fgetc(stdin);

    /* restore the original terminal attributes */
    tcsetattr(fileno(stdin), TCSANOW, &orig_term_attr);

    return character;
}



int main1( int argc, char** argv)
{
    int n = 0;
    char buf_send[BUFF_SIZE];

    int key = 0;

    string s = "0\n";


    int cport_nr = api_uart_open();

    bool start = false;

    if( cport_nr != -1 )
    {
        while(1)
        {
            key = getkey();
            if( key == 's' )
            {
                start = !start;
            }

            if( key == 'f' )
                break;

            if( start )
            {
                s = "100\n";
                strcpy(buf_send, s.c_str());
                printf("just sent: %s\n", buf_send);
            }
            else
            {
                s = "0\n";
                strcpy(buf_send, s.c_str());
                printf("just sent: %s\n", buf_send);
            }

            api_uart_write(cport_nr, buf_send);

            usleep(2000000);  /* sleep for 1 Second */
        }

    }

    return(0);
}

int main( int argc, char** argv)
{
    int n = 0;
    char buf_send[BUFF_SIZE];

    char buf_recv[BUFF_SIZE];

    string s = argv[1];
    s = s + "\n";

    strcpy(buf_send, s.c_str());

    int cport_nr = api_uart_open();

    if( cport_nr != -1 )
    {
        while(1)
        {
            api_uart_write(cport_nr, buf_send);

            if( buf_send[0] == 's' )
            {
            for(int i=0;i<BUFF_SIZE-1;i++)
            buf_send[i]=buf_send[i+1];
            buf_send[BUFF_SIZE-1]='\0';
                printf("just sent: %s\n", buf_send);
            }

            if( buf_send[0] == 'g' )
            {
                printf("sent: %s\n", buf_send);
                usleep(5000);

                n = api_uart_read( cport_nr, buf_recv);

                if(n > 0)
                {
                    buf_recv[n] = 0;

                    printf("recv: %s\n", buf_recv);
                }
                for( int i = 0; i < BUFF_SIZE; i++ )
                    buf_recv[i] = 0;
            }

            usleep(2000000);  /* sleep for 1 Second */
        }

    }

    return(0);
}
