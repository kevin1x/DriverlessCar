#include "api_i2c_pwm.h"
#include "api_uart.h"
#include <iostream>

using namespace std;

FILE *logUart;
int cport_nr;
char buf_send[BUFF_SIZE], buf_recv[BUFF_SIZE];
int logId = 0;

void loadKey(char ch) {
    int toSend = (ch - '0') * 100;
    sprintf(buf_send, "%d\n", toSend);
	api_uart_write(cport_nr, buf_send);
	fprintf(stderr, "sent %d\n", toSend);
	fprintf(logUart, "Event write: %d\n", toSend);
}

int main(int argc, char **argv) {
    logUart = fopen("logUart.txt", "w");
    PCA9685 *pca9685 = new PCA9685() ;
    api_pwm_pca9685_init( pca9685 );

    int dir = 0, throttle_val = 32;
    double theta = 0;
    int current_state = 0;
    char key = 0;
    dir = DIR_REVERSE;
    theta = 0;
    if (pca9685->error >= 0)
        api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
    cport_nr = api_uart_open();

    if( cport_nr == -1 ) {
        cerr<< "Error: Canot Open ComPort";
        return -1;
    }
    fprintf(stderr, "port %d\n", cport_nr);
    bool start = false;
    while ( true )
    {
        key = getkey();
        if( key == 's') {
            start = !start;
            if (start) {throttle_val = 32; fprintf(stderr, "ON\n"); api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );}
            else fprintf(stderr, "OFF\n");
        }
        if( key == 'f') break;
        
        if (start) {
            if ('0' <= key && key <= '9')
                loadKey(key);
            int n = api_uart_read(cport_nr, buf_recv);
            if (n > 0 && logUart != NULL) fprintf(logUart, "%s\n", buf_recv);
        }
        else {
            theta = 0;
            throttle_val = 0;
			api_uart_write(cport_nr, buf_send);
			api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
            sleep(1);
        }
    }
    theta = 0;
    throttle_val = 0;
    api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
    return 0;
}
