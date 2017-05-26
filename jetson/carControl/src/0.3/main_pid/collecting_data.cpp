#include <fstream> 
#include "api_kinect_cv.h"
// api_kinect_cv.h: manipulate openNI2, kinect, depthMap and object detection
#include "api_lane_detection.h"
// api_lane_detection.h: manipulate line detection, finding lane center and vanishing point
#include "api_i2c_pwm.h"
#include "api_uart.h"
#include <iostream>

using namespace openni;

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
#define VIDEO_FRAME_WIDTH 640
#define VIDEO_FRAME_HEIGHT 480

/// Get depth Image or BGR image from openNI device
/// Return represent character of each image catched
char analyzeFrame(const VideoFrameRef& frame, Mat& depth_img, Mat& color_img) {
    DepthPixel* depth_img_data;
    RGB888Pixel* color_img_data;

    int w = frame.getWidth();
    int h = frame.getHeight();

    depth_img = Mat(h, w, CV_16U);
    color_img = Mat(h, w, CV_8UC3);
    Mat depth_img_8u;
	
    switch (frame.getVideoMode().getPixelFormat())
    {
        case PIXEL_FORMAT_DEPTH_1_MM: return 'm';
        case PIXEL_FORMAT_DEPTH_100_UM:

            depth_img_data = (DepthPixel*)frame.getData();

            memcpy(depth_img.data, depth_img_data, h*w*sizeof(DepthPixel));

            normalize(depth_img, depth_img_8u, 255, 0, NORM_MINMAX);

            depth_img_8u.convertTo(depth_img_8u, CV_8U);

            return 'd';
        case PIXEL_FORMAT_RGB888:
            color_img_data = (RGB888Pixel*)frame.getData();

            memcpy(color_img.data, color_img_data, h*w*sizeof(RGB888Pixel));

            cvtColor(color_img, color_img, COLOR_RGB2BGR);
		
            return 'c';
        default:
            printf("Unknown format\n");
            return 'u';
    }
}

/// Return angle between veritcal line containing car and destination point in degree
double getTheta(Point car, Point dst) {
    if (dst.x == car.x) return 0;
    if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
    double pi = acos(-1.0);
    double dx = dst.x - car.x;
    double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}

int cport_nr; // port id of uart.
char buf_send[BUFF_SIZE]; // buffer to store and recive controller messages.

/// Write speed to buffer
void setThrottle(int speed) {
	if (speed>=0)
    sprintf(buf_send, "f%d\n", speed);
	else { 
		speed=-speed;
		sprintf(buf_send, "b%d\n", speed);
	}
}


///////// utilitie functions  ///////////////////////////

int main( int argc, char* argv[] ) {

//init file
    ofstream myfile;
	myfile.open ("data.txt");			//open file to write data

/// Init openNI ///
    Status rc;
    Device device;

    VideoStream depth, color;
    rc = OpenNI::initialize();
    if (rc != STATUS_OK) {
        printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
        return 0;
    }
    rc = device.open(ANY_DEVICE);
    if (rc != STATUS_OK) {
        printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
        return 0;
    }
    if (device.getSensorInfo(SENSOR_DEPTH) != NULL) {
        rc = depth.create(device, SENSOR_DEPTH);
        if (rc == STATUS_OK) {
            VideoMode depth_mode = depth.getVideoMode();
            depth_mode.setFps(30);
            depth_mode.setResolution(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT);
            depth_mode.setPixelFormat(PIXEL_FORMAT_DEPTH_100_UM);
            depth.setVideoMode(depth_mode);

            rc = depth.start();
            if (rc != STATUS_OK) {
                printf("Couldn't start the color stream\n%s\n", OpenNI::getExtendedError());
            }
        }
        else {
            printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
        }
    }

    if (device.getSensorInfo(SENSOR_COLOR) != NULL) {
        rc = color.create(device, SENSOR_COLOR);
        if (rc == STATUS_OK) {
            VideoMode color_mode = color.getVideoMode();
            color_mode.setFps(30);
            color_mode.setResolution(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT);
            color_mode.setPixelFormat(PIXEL_FORMAT_RGB888);
            color.setVideoMode(color_mode);

            rc = color.start();
            if (rc != STATUS_OK)
            {
                printf("Couldn't start the color stream\n%s\n", OpenNI::getExtendedError());
            }
        }
        else {
            printf("Couldn't create color stream\n%s\n", OpenNI::getExtendedError());
        }
    }
    
    VideoFrameRef frame;
    VideoStream* streams[] = {&depth, &color};
/// End of openNI init phase ///
    
/// Init video writer and log files ///   
    bool is_save_file = true; // set is_save_file = true if you want to log video and i2c pwm coeffs.
    VideoWriter depth_videoWriter;	
    VideoWriter color_videoWriter;
    VideoWriter gray_videoWriter;
     
    string gray_filename = "gray.avi";
	string color_filename = "color.avi";
	string depth_filename = "depth.avi";
	
	Mat depthImg, colorImg, grayImage;
	int codec = CV_FOURCC('D','I','V', 'X');
	int video_frame_width = VIDEO_FRAME_WIDTH;
    int video_frame_height = VIDEO_FRAME_HEIGHT;
	Size output_size(video_frame_width, video_frame_height);

   
	if(is_save_file) {
	    gray_videoWriter.open(gray_filename, codec, 8, output_size, false);
        color_videoWriter.open(color_filename, codec, 8, output_size, true);
        //depth_videoWriter.open(depth_filename, codec, 8, output_size, false);
       
	}
/// End of init logs phase ///
    int dir = 0;
    int throttle_val = 0;
    double theta = 0;
    int current_state = 0;
    char key = 0;
    int set_throttle_val = 0;
    //=========== Init  =======================================================

    ////////  Init PCA9685 driver   ///////////////////////////////////////////

    PCA9685 *pca9685 = new PCA9685() ;
    api_pwm_pca9685_init( pca9685 );

    if (pca9685->error >= 0)
        api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );

    /////////  Init UART here   ///////////////////////////////////////////////

	cport_nr = api_uart_open();

    if( cport_nr == -1 ) {
        cerr<< "Error: Canot Open ComPort";
        return -1;
    }

    
    bool running = false, started = false, stopped = false;
    
    int turning , move;
    int count_frame = -1;
    int last_throttle = throttle_val;
    while ( true )
    {
        ++count_frame;
        
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

        if( running )
        {
			//// Check PCA9685 driver ////////////////////////////////////////////
            if (pca9685->error < 0)
            {
                cout<< endl<< "Error: PWM driver"<< endl<< flush;
                break;
            }
			if (!started)
			{
    			fprintf(stderr, "ON\n");
			    started = true; stopped = false;
				throttle_val = set_throttle_val;
				setThrottle(throttle_val);
				api_uart_write(cport_nr, buf_send);
			}


        




            
            ////////  Get input image from camera   //////////////////////////////
            int readyStream = -1;
		    rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
		    if (rc != STATUS_OK)
		    {
		        printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
		        break;
		    }

		    switch (readyStream)
		    {
		        case 0:
		            // Depth
		            depth.readFrame(&frame);
		            break;
		        case 1:
		            // Color
		            color.readFrame(&frame);
		            break;
		        default:
		            printf("Unxpected stream\n");
		    }
		    char recordStatus = analyzeFrame(frame, depthImg, colorImg);
		   
            ////////// Detect Center Point ////////////////////////////////////
            if (recordStatus == 'c') {
                
                cvtColor(colorImg, grayImage, CV_BGR2GRAY);
               
                
            }
            ///////  Your PID code here  //////////////////////////////////////////
            
            switch (key)
			{		
			    case 'k':			//right
			        if (theta<200) theta += 40;				       			        
			        break;
			    case 'h':			//left
			        if (theta>-200) theta -= 40;
			        break;
			    case 'u':			//up
				    if (throttle_val<50) 
				    {
				        last_throttle = throttle_val;
					    throttle_val += 2;
					    setThrottle(throttle_val);
					    api_uart_write(cport_nr, buf_send);						
				    }
			        break;
			    case 'j':			//down
			        if (throttle_val>-50)
				    {
    				    last_throttle = throttle_val;
					    throttle_val -= 2;
					    setThrottle(throttle_val);
					    api_uart_write(cport_nr, buf_send);						
				    }
			        break;    
			    default :
			        break;    
			}
                
            if (theta < 0) turning = 1;                           //left
            else if (theta > 0) turning = 2;                      //right
            else turning = 0;                                    //not left or right
               
            if (last_throttle < throttle_val) move = 1;          //down
            else if (throttle_val != 0) move = 2;                //up
            else move = 0;                                       //not up or down

                
            myfile << count_frame << " " << move << " " << turning << " " << throttle_val << " " << theta << endl ;


			
            int pwm2 = api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
            cout<< endl<< "Frame: "<<count_frame<< "; Theta: "<< theta<< "; Throttle: "<< throttle_val<< flush;

			

            if (recordStatus == 'c' && is_save_file) {

                if (!colorImg.empty())
			        color_videoWriter.write(colorImg);
			    if (!grayImage.empty())
			        gray_videoWriter.write(grayImage); 
            }
            if (recordStatus == 'd' && is_save_file) {
                if (!depthImg.empty())
                   depth_videoWriter.write(depthImg);
            }

            
            if( key == 27 ) break;
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
    //////////  Release //////////////////////////////////////////////////////
	if(is_save_file)
    {
        gray_videoWriter.release();
        color_videoWriter.release();
        //depth_videoWriter.release();
	}
	
	//release file
	myfile.close();
    return 0;
}














