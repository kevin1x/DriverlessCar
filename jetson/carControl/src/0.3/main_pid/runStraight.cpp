/**
    Run car with theta = 0;
    If your car didn't run quite straight, you may have to config STEERINGs' constants in api_i2c_pwm.h
**/

#include "api_i2c_pwm.h"
#include "api_uart.h"
#include <iostream>

using namespace std;

int cport_nr;
char buf_send[BUFF_SIZE];

void setThrottle(int speed) {
	if (speed>=0)
    sprintf(buf_send, "f%d\n", speed);
	else { 
		speed=-speed;
		sprintf(buf_send, "b%d\n", speed);
	}
}

int main(int argc, char **argv) {
    PCA9685 *pca9685 = new PCA9685() ;
    api_pwm_pca9685_init( pca9685 );

    int dir = 0, throttle_val = 0, set_throttle_val = 28;
    double theta = 0;
    if (argc > 1) sscanf(argv[1], "%d", &set_throttle_val);
    throttle_val = set_throttle_val;
    fprintf(stderr, "Initial speed: %d\n", set_throttle_val);
    int current_state = 0;
    char key = 0;
    dir = DIR_REVERSE;
    if (pca9685->error >= 0)
        api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
    cport_nr = api_uart_open();

    if( cport_nr == -1 ) {
        cerr<< "Error: Canot Open ComPort";
        return -1;
    }
    fprintf(stderr, "Press s to start, s again to switch start / stop and f to finish.\n");
    bool running = false, started = false, stopped = false;
    
    while ( true )
    {
        key = getkey();
        if( key == 's') {
            running = !running;
        }
        if( key == 'f') {
            fprintf(stderr, "End process.\n");
            theta = 0;
            throttle_val = 0;
            setThrottle(throttle_val);
            api_uart_write(cport_nr, buf_send);
	        api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
            break;
        }
        
        if (running) {
            if (!started) {
                fprintf(stderr, "ON\n");
                started = true; stopped = false;
                theta = 0;
                throttle_val = set_throttle_val;
                setThrottle(throttle_val);
	    		api_uart_write(cport_nr, buf_send);
	    		api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
	        }
        }
        else {
            theta = 0;
            throttle_val = 0;
            if (!stopped) {
                fprintf(stderr, "OFF\n");
                stopped = true; started = false;
                setThrottle(throttle_val);
                api_uart_write(cport_nr, buf_send);
			}
			api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
            sleep(1);
        }
    }
    
    theta = 0;
            throttle_val = 0;
            setThrottle(throttle_val);
            api_uart_write(cport_nr, buf_send);
	        api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
    return 0;
}
