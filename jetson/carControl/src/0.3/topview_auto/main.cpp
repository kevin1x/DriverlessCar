#include <bits/stdc++.h>
#include <opencv/cv.hpp>

#include "IPM.h"
#include "xla.h"

using namespace std;
using namespace cv;

void draw_line(Mat &destination, Line line, int shift_x = 0) {
    int height = destination.rows;
    Point A = Point(line.intersect(0), 0);
    Point B = Point(line.intersect(height), height);
    A.x += shift_x;
    B.x += shift_x;
    cv::line(destination, A, B, Scalar(0, 255, 0), 3, 8);
}

double go(Mat &source, int x, int y) {
    int width = source.cols;
    int height = source.rows;
    static Mat destination = source.clone();

    cerr << "x, y = " << x << ',' << y << endl;
    vector<Point2f> origPoints;
    origPoints.push_back( Point2f(0, height) );
    origPoints.push_back( Point2f(width, height) );
    origPoints.push_back( Point2f(width - x - 1, y) );
    origPoints.push_back( Point2f(x, y) );
    vector<Point2f> dstPoints;
    dstPoints.push_back( Point2f(0, height) );
    dstPoints.push_back( Point2f(width, height) );
    dstPoints.push_back( Point2f(width, 0) );
    dstPoints.push_back( Point2f(0, 0) );
    IPM ipm(Size(width, height), Size(width, height), origPoints, dstPoints);
    ipm.applyHomography(source, destination);
    Rect left_ROI(0, 0, width / 2, height);
    Rect right_ROI(width / 2, 0, width / 2, height);
    Mat left_image = destination(left_ROI).clone();
    Mat right_image = destination(right_ROI).clone();
    Line left_line = process_frame(left_image);
    Line right_line = process_frame(right_image);

    draw_line(destination, left_line);
    draw_line(destination, right_line, width / 2);
    imshow("left", left_image);
    imshow("right", right_image);
    imshow("destination", destination);

    double diff = fabs(left_line.angle() - right_line.angle());
    if (diff < 0.05) {
        cerr << "Diff = " << diff << endl;
        //waitKey(0);
    } else {
        //waitKey(100);
    }

    //waitKey(1);
    return diff;
}


int main(int argc, char **argv) {
    string filename = argv[1];
    VideoCapture capture(filename);
    Mat source, destination;
    if( !capture.isOpened()) throw "Error reading video";

    capture >> source;
    cvtColor(source, source, CV_BGR2GRAY);

    int width = capture.get(3);
    int height = capture.get(4);

    cerr << "width = " << width << endl;
    cerr << "height = " << height << endl;

    int best_x = 0, best_y = 0;
    double best_diff = 99;

    for (int x = width / 2 - 150; x < width / 2 - 100; ++x) {
        //for (int y = 300;;) {
        for (int y = 280; y < 330; ++y) {
            double cur = go(source, x, y);
            if (best_diff > cur) {
                best_diff = cur;
                best_x = x;
                best_y = y;
            }
        }
    }

    ofstream fout("params.txt");
    fout << 0 << ' ' << height << endl;
    fout << width << ' ' << height << endl;
    fout << width - best_x - 1 << ' ' << best_y << endl;
    fout << best_x << ' ' << best_y << endl;
    fout.close();

    cerr << "best_diff = " << best_diff << endl;
    go(source, best_x, best_y);
    waitKey(0);
    return 0;
}