#include "api_i2c_pwm.h"
#include "../uart/api_uart.h"
#include <iostream>

using namespace std;
char buf_send[BUFF_SIZE];

void setControl(int speed,int angle)
{

	if (speed>=0) sprintf(buf_send, "f%d\n", speed);
	else
	{ 
		speed=-speed;
		sprintf(buf_send, "b%d\n", speed);
	}
}

int main( int argc, char** argv )
{
    PCA9685 *pca9685 = new PCA9685() ;

    api_pwm_pca9685_init( pca9685 );

    int dir = 0, throttle_val = 0, pwm2 = 390;
    double theta = 0;
    int current_state = 0;

    // 27 is the ESC key

    cout<< endl<< "Hit ESC key to exit"<< flush;

    char key = 0;

    int cport_nr = api_uart_open();
    if( cport_nr == -1 )
    {
        cerr<< "Error: Canot Open ComPort";
        return -1;
    }

    while(pca9685->error >= 0)
    {
        key = getkey();

        switch (key)
        {       
        case 'v':
            theta += 10;
            break;
        case 'b':
            theta -= 10;
            break;

        case 'j':
			if (throttle_val<50) 
			{
				throttle_val += 10;
        		setControl(throttle_val,theta);
				api_uart_write(cport_nr, buf_send);
        		
			}
            break;

        case 'g':
            if (throttle_val>-50)
			{
				throttle_val -= 10;
        		setControl(throttle_val,theta);
				api_uart_write(cport_nr, buf_send);
        		
			}
            break;        

        case 'h':
            dir++;
            if(dir > 2 ) dir = 0;   
            break;

        }

        if( key == 27 )
            break;
        api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );

        cout<< endl<< "Theta: "<< theta<< "; Throttle: "<< throttle_val<< flush;
			cout<< endl<< "Data: "<< buf_send<< flush;

        sleep(1);
    }

    api_pwm_pca9685_release( pca9685 );

    return 0;
}






