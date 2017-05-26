#ifndef XLA_H
#define XLA_H

#include <bits/stdc++.h>
#include <opencv/cv.hpp>

#include "./lib/api_lane_detection.h"
#include "./lib/IPM.h"
#include "./lib/line_segment.h"

using namespace std;
using namespace cv;

const double ANGLE_THRESHOLD = PI / 8;
const double LENGTH_THRESHOLD = 50;
const double DIST_THRESHOLD = 800;

class TopviewConverter {
public:
    IPM ipm;

    TopviewConverter() {}

    TopviewConverter(vector<Point2f> &origPoints, int width = 640, int height = 480) {
        vector<Point2f> dstPoints;
        dstPoints.push_back(Point2f(0, height));
        dstPoints.push_back(Point2f(width, height));
        dstPoints.push_back(Point2f(width, 0));
        dstPoints.push_back(Point2f(0, 0));
        ipm = IPM(Size(width, height), Size(width, height), origPoints, dstPoints);
    }

    void convert(const Mat &inputImg, Mat &outputImg) {
        ipm.applyHomography(inputImg, outputImg);
    }
};

class XLA {
private:
    static const int LOCAL_FRAME_WIDTH = 640/3;
    static const int LOCAL_FRAME_HEIGHT = 480/3;

    template<typename T> T sqr(T x) {
        return x * x;
    }
    template<typename T> int sign(T x) {
        if (x == 0) return 0;
        return x < 0 ? -1 : 1;
    }

    TopviewConverter bird;

public:
    XLA() {}
    XLA(string file_path) { read_topview_params(file_path); }
    void run_hough_transform(const Mat &binary_frame, vector<Vec4i> &lines);
    void show_angle(Mat &output, double rad);//ve goc arrow len image
    void draw_lines(Mat&output, vector<Line> &lines); //ve lines len image
    double get_steering_angle(vector<Line> &lines); //tinh trung binh goc theo lines
    int get_average_intensity(Mat &gray_frame, vector<Point> &contour);
    void lets_be_handsome(Mat &bgr_frame, Mat &gray_frame, Mat &topview_frame, Mat &binary_frame);
    Line process_frame(Mat &bgr_frame, VideoWriter &bgr_writer, int &topview_height, int &topview_width); // tra ve approximation-line theo frame
    double adjust_angle(Line previous_line, Line current_line, int topview_height, int topview_width); //chinh lai goc mot ti
    void convert_to_topview(const Mat &inputImg, Mat &outputImg); // ham converttopview cua em 
    void read_topview_params(string file_path); //doc tham so topview tu file cua em
    vector<Line> process_one_side(Mat &binary_frame); //tinh + loc ra approximation line theo mot lane
    vector<Line> get_all_lines(Mat &binary_frame); // giong ham  run_hough_transform nhung tra ve vector<Line>
    double get_sum_length(vector<Line> &lines); //tinh tong do dai cua cac Line trong lines
    vector<Line> filter_lines(vector<Line> &d); //loc ra nhung line gan voi lane
    Mat color_filter(Mat frame); //loc mau
    void do_cut(const Mat &binary_image, Line split_line, Mat &left_image, Mat &right_image); //cat ra 2 nua left-right
    bool invalid_split(Line split_line, int x, int y); //check xem split_line co cat tren duoi hay khong
    void filter_binary_image(Mat &topview, Mat &binary); //loc hinh thang topview
    void filter_lines_angle(vector<Line> &d);
};


#endif //XLA_H
