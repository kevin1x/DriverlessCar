//#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "stdio.h"
#include <iomanip>
#include <OpenNI.h>
#include <termios.h>
#include <fstream>
#include <iostream>


#define GRAY_LEVEL_PER_MINIMETER  20
#define DIST_MIN    50      // 1000 mm
#define DIST_MAX    200     // 4000 mm
#define SLICE_DEPTH 10      // 200  mm
#define SLICE_NB    15
#define ROI_PLAN    75     // 2500 mm



using namespace cv;
using namespace std;
using namespace openni;

void
api_kinect_cv_disparity2color( const Mat& gray,
                   Mat& rgb,
                   double maxDisp=-1.f,
                   float S=1.f, float V=1.f );

float
api_kinect_cv_get_max_disparity( VideoCapture& capture );

int
api_kinect_cv_get_images(
        VideoCapture &capture,
        Mat &depthMap,
        Mat &grayImage
        );

struct api_kinect_cv_obj_t
{
    int x;
    int y;
    int width;
    int height;
    int depth;
};


void
api_kinect_cv_get_obtacle_rect(Mat& depthMap,
                               vector< Rect > &output_boxes,
                               Rect &roi,
                               int lower_bound = DIST_MIN,
                               int upper_bound = DIST_MIN + 2*SLICE_DEPTH
                           );

void
api_kinect_cv_center_rect_gen(
        vector< Rect > &rects,
        int frame_width,
        int frame_height
        );
