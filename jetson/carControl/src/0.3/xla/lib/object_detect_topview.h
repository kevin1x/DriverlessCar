#ifndef OBJECTTOPVIEW_H
#define OBJECTTOPVIEW_H

#include <opencv/cv.hpp>

#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;
using namespace cv;

Mat Topview_only(Mat frame); //convert to topview (BGR -> BGR)
Mat getObjectMat(Mat frame); //detect object (BGR -> Topview_BINARY)

#endif //OBJECTTOPVIEW_H
