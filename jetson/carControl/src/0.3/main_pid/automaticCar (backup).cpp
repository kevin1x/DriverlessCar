/**
    This code runs our car automatically and log video, controller (optional)
    Line detection method: Canny
    Targer point: vanishing point
    Control: pca9685
    
    You should understand this code run to image how we can make this car works from image processing coder's perspective.
    Image processing methods used are very simple, your task is optimize it.
    Besure you set throttle val to 0 before end process. If not, you should stop the car by hand.
    In our experience, if you accidental end the processing and didn't stop the car, you may catch it and switch off the controller physically or run the code again (press up direction button then enter).
**/
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
    bool is_save_file = false; // set is_save_file = true if you want to log video and i2c pwm coeffs.
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

   	FILE *thetaLogFile; // File creates log of signal send to pwm control
	if(is_save_file) {
	    gray_videoWriter.open(gray_filename, codec, 8, output_size, false);
        color_videoWriter.open(color_filename, codec, 8, output_size, true);
        //depth_videoWriter.open(depth_filename, codec, 8, output_size, false);
        thetaLogFile = fopen("thetaLog.txt", "w");
	}
/// End of init logs phase ///

    int dir = 0, throttle_val = 0;
    double theta = 0;
    int current_state = 0;
    char key = 0;

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

    /// Init MSAC vanishing point library
    MSAC msac;
    cv::Rect roi1 = cv::Rect(0, VIDEO_FRAME_HEIGHT*3/4,
                            VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT/4);

    api_vanishing_point_init( msac );

    ////////  Init direction and ESC speed  ///////////////////////////
    dir = DIR_REVERSE;
    int set_throttle_val = 0;
    throttle_val = 0;
    theta = 0;
    
    // Argc == 2 eg ./test-autocar 27 means initial throttle is 27
    if(argc == 2 ) set_throttle_val = atoi(argv[1]);
    fprintf(stderr, "Initial throttle: %d\n", set_throttle_val);
    int frame_width = VIDEO_FRAME_WIDTH;
    int frame_height = VIDEO_FRAME_HEIGHT;
    Point carPosition(frame_width / 2, frame_height);
    /** 
        We chose car position is center of bottom of the image.
        Indeed, it's a rectangle. We used only a single point to make it simple and enough-to-work.
    **/
    Point prvPosition = carPosition;
    //============ End of Init phase  ==========================================

    //============  PID Car Control start here. You must modify this section ===

    bool running = false, started = false, stopped = false;

    double st = 0, et = 0, fps = 0;
    double freq = getTickFrequency();


    bool is_show_cam = true;
	int count_s,count_ss;
    while ( true )
    {
        Point center_point(0,0);

        st = getTickCount();
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
                /**
                    Status = BGR Image. Since we developed this project to run a sample, we only used information of gray image.
                **/
                cvtColor(colorImg, grayImage, CV_BGR2GRAY);
                // api_get_lane_center( grayImage, center_point, true);
                api_get_vanishing_point( grayImage, roi1, msac, center_point, is_show_cam,"Canny");
                /**
                    Note 1: The real signal that we send to controller is angle, to find sufficient wheels angle
                        you can find target point (lance center, vanishing point,...) then determine wheels angle
                    Note 2: We provide 2 methods to find target point: lane center (api_get_lane_center) and MSAC vanishing point (api_get_vanishing_point)
                        and many methos for line detection: Canny, Sobels, Prewitt, Roberts, Wavelet,..
                        All line detect methods is just parameters, see "Canny" above.
                        We won't go to details how these methods work for 2 reasons:
                            1. You are asked to find target point in elimination contest
                            2. You may have your own implementations for those works. If so, optimizing your idea is much more better than understanding our enough-to-work implementations.
                **/
                if (center_point.x == 0 && center_point.y == 0) center_point = prvPosition;
                /**
                    Note 1: For some reasons, we consider point (0, 0) is invalid point.
                        We did a trick that if we found an invalid target point, we use our previous target point.
                        Again, it's your own choice
                    Note 2: If you want to use -1 or any other negative value as invalid value, be sure you won't get data type overflow.
                 **/
                prvPosition = center_point;
                double angDiff = getTheta(carPosition, center_point);
                theta = (-angDiff * 3);
                /**
                    Theta is angle between center of image bottom and target point CLOCKWISE
                    Note 1: I calculated theta in paper, it is theta has the same sign with angDiff in my eyes.
                         But my camera catched reverse image so theta and angDiff have opposite signs, you should check what direction it catched to find correct sign of theta.
                    Note 2: For this specific car, max left and right wheels' angle is about 20 degree
                        We scaled it into [-200, 200] which means if you want to move right 2 degree, set theta = 2 * 20 or -2 * 20
                        Idk other wheels direction, so try to change theta's sign if you think something is wrong.
                    Note 3: First theta is used as an integer value, but finally we used double to enhance smoothness
                        An obvious disadvantage is slower speed, you can change theta into integer to speed up.
                    Note 4: Theta and the magic 3 number (theta = 0.3 * angDiff in real - for smoothness) is my own choice
                        You should optimize it yourself, we made this just enough to run sample.
                **/
            }
            ///////  Your PID code here  //////////////////////////////////////////
			
            int pwm2 = api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
            et = getTickCount();
            fps = 1.0 / ((et-st)/freq);
            cerr << "FPS: "<< fps<< '\n';

            if (recordStatus == 'c' && is_save_file) {
                // 'Center': target point
                // pwm2: STEERING coefficient that pwm at channel 2 (our steering wheel's channel)
                fprintf(thetaLogFile, "Center: [%d, %d]\n", center_point.x, center_point.y);
                fprintf(thetaLogFile, "pwm2: %d\n", pwm2);
                cv::circle(colorImg, center_point, 4, cv::Scalar(0, 255, 255), 3);
                if (!colorImg.empty())
			        color_videoWriter.write(colorImg);
			    if (!grayImage.empty())
			        gray_videoWriter.write(grayImage); 
            }
            if (recordStatus == 'd' && is_save_file) {
                if (!depthImg.empty())
                   depth_videoWriter.write(depthImg);
            }

            //////// using for debug stuff  ///////////////////////////////////
            if(is_show_cam) {
                if(!grayImage.empty())
                    imshow( "gray", grayImage );
                waitKey(10);
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
        fclose(thetaLogFile);
	}
    return 0;
}


