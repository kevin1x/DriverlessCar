/**
    This code is to test steering coefficient, you can use this code to find suitable coefficients by hand
    Press w or a to increase pwm2, wheels reach no more then it's STEERING_MAX_LEFT
    Press s or d to decrease pwm2, wheesl reach no more then it's STEERING_MAX_RIGHT
        You should set STEERING_NEUTRAL = (STEERING_MAX_LEFT + STEERING_MAX_RIGHT) / 2.
    Controller coefficients are just approximations. When steering wheels reach their limit, it may sometimes work, sometimes not.
    If you run code ./run-straight, the car run 3 meters without error of 3cm, it's acceptable.
**/
#include "api_i2c_pwm.h"
#include "api_uart.h"
#include <iostream>

using namespace std;

int main( int argc, char** argv )
{
    PCA9685 *pca9685 = new PCA9685() ;

    api_pwm_pca9685_init( pca9685 );

    int dir = 0, pwm2 = 350;
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

            case 'w':
                pwm2 += 10; break;
            case 's':
                pwm2 -= 10; break;
            case 'a':
                pwm2++; break;
            case 'd':
                pwm2--; break;
        }

        if( key == 27 )
            break;
	    fprintf(stderr, "pwm2 = %d\n", pwm2);
	    pca9685->setPWM(STEERING_CHANNEL2,0, pwm2);
        sleep(1);
    }

    api_pwm_pca9685_release( pca9685 );

    return 0;
}






