#ifndef LINETOPVIEW_H
#define LINETOPVIEW_H
#include <opencv/cv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;
using namespace cv;

Mat Detect_color(Mat frame); //white color filter (BGR -> BINARY)
Mat Topview_transform(Mat frame); //white color filter + topview_transform (BGR -> TopviewBinary)

#endif
