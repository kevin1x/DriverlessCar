#include "xla.h"
#define SHOW_OUTPUT
//#define WRITE_VIDEO

const int LOCAL_FRAME_WIDTH = 640;
const int LOCAL_FRAME_HEIGHT = 480;

void convert_to_topview(const Mat &inputImg, Mat &outputImg) {
    int height = inputImg.rows;
    int width = inputImg.cols;

    vector<Point2f> origPoints;
    origPoints.push_back(Point2f(0, height));
    origPoints.push_back(Point2f(width, height));
    origPoints.push_back(Point2f(width/2+120, 300));
    origPoints.push_back(Point2f(width/2-120, 300));

    vector<Point2f> dstPoints;
    dstPoints.push_back(Point2f(0, height));
    dstPoints.push_back(Point2f(width, height));
    dstPoints.push_back(Point2f(width, 0));
    dstPoints.push_back(Point2f(0, 0));

    IPM ipm(Size(width, height), Size(width, height), origPoints, dstPoints);
    ipm.applyHomography(inputImg, outputImg);
}

void convert_back(Mat inputImg, Mat &outputImg) {
    int height = inputImg.rows;
    int width = inputImg.cols;

    vector<Point2f> origPoints;
    origPoints.push_back(Point2f(0, height));
    origPoints.push_back(Point2f(width, height));
    origPoints.push_back(Point2f(width/2+120, 300));
    origPoints.push_back(Point2f(width/2-120, 300));

    vector<Point2f> dstPoints;
    dstPoints.push_back(Point2f(0, height));
    dstPoints.push_back(Point2f(width, height));
    dstPoints.push_back(Point2f(width, 0));
    dstPoints.push_back(Point2f(0, 0));

    IPM ipm(Size(width, height), Size(width, height), dstPoints, origPoints);
    ipm.applyHomography(inputImg, outputImg);
}

void run_hough_transform(const Mat &binary_frame, vector<Vec4i> &lines) {
    int houghThreshold = 100;
    HoughLinesP(binary_frame, lines, 2, CV_PI/90, houghThreshold, 10,30);
}

#define sqr(x) ((x)*(x))

void show_angle(Mat &output, double rad) {
    Point bottom(output.cols / 2, output.rows);
    Point to(-cos(rad) * 100 + output.cols / 2, output.rows - sin(rad) * 100);
    arrowedLine(output, bottom, to, Scalar(50, 50, 255), 3, 8);
}

void draw_lines(Mat&output, vector<Vec4i> &lines) {
    printf("Hough found %d lines\n", (int)lines.size());
    for(size_t i = 0; i < lines.size(); i++) {
        line(output, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
    }
}

double get_steering_angle(vector<Vec4i> &lines) { //radian
    double sum_angle = 0;
    double sum_weight = 0;

    for(size_t i = 0; i < lines.size(); i++) {
        double dist = sqrt(sqr(lines[i][0] - lines[i][2]) + sqr(lines[i][1] - lines[i][3]));
        double angle = atan2(lines[i][3] - lines[i][1], lines[i][2] - lines[i][0]);
        if (angle < 0) {
            angle = PI + angle;
        }
        sum_weight += dist;
        sum_angle += dist * angle;
    }

    sum_angle /= sum_weight;
    return sum_angle;
}

Line process_frame(Mat &bgr_frame, VideoWriter &bgr_writer) {
    static Mat gray_frame;
    static Mat topview_frame;
    static Mat binary_frame;

    convert_to_topview(bgr_frame, topview_frame);
    //resize(topview_frame, topview_frame, Size(LOCAL_FRAME_WIDTH, LOCAL_FRAME_HEIGHT));
    cvtColor(topview_frame, gray_frame, CV_BGR2GRAY);
    
    static Mat element = cv::getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    edgeProcessing(gray_frame, binary_frame, element, "Wavelet");
    
    vector<Vec4i> lines;
    run_hough_transform(binary_frame, lines);
    vector<Line> d;
    for (int i = 0; i < lines.size(); ++i) {
        d.push_back(Line(Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3])));
    }
    //lines = filter_lines(lines);
    Mat hough_frame = topview_frame.clone();
    double steering_angle = get_steering_angle(lines);
    Line res = approximate(d, steering_angle);

#ifdef WRITE_VIDEO
    draw_lines(hough_frame, lines);
    cerr << "res = " << res.a << ' ' << res.b << endl;
    /*
    cerr << "res = " << res.P.x << ' ' << res.P.y << ' ' << res.Q.x << ' ' << res.Q.y << endl;
    cerr << "intersect = " << res.intersect(240) << endl;
    */
    Point A = Point(res.intersect(0), 0);
    Point B = Point(res.intersect(480), 480);
    cerr << "A = " << A.x << ' ' << A.y << endl;
    cerr << "B = " << B.x << ' ' << B.y << endl;
    line(hough_frame, A, B, Scalar(0, 255, 0), 3, 8);

    show_angle(hough_frame, steering_angle);
#endif
    //convert_back(hough_frame, hough_frame);

#ifdef SHOW_OUTPUT
    imshow("bgr", bgr_frame);
    imshow("topview", topview_frame);
    imshow("binary", binary_frame);
    imshow("hough", hough_frame);
    waitKey(30);
#endif

#ifdef WRITE_VIDEO
    cerr << "bgr_writer " << hough_frame.rows << ' ' << hough_frame.cols << endl;
    bgr_writer << hough_frame;
#endif

    //return steering_angle * 180 / PI - 90;
    return res;
}

Line process_frame(Mat &topview_frame) {
    static Mat gray_frame;
    static Mat binary_frame;

    //convert_to_topview(bgr_frame, topview_frame);
    //resize(topview_frame, topview_frame, Size(LOCAL_FRAME_WIDTH, LOCAL_FRAME_HEIGHT));
    if (topview_frame.channels() == 3){
        cvtColor(topview_frame, gray_frame, CV_BGR2GRAY);
    } else {
        gray_frame = topview_frame.clone();
    }
    
    static Mat element = cv::getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    edgeProcessing(gray_frame, binary_frame, element, "Wavelet");
    
    vector<Vec4i> lines;
    run_hough_transform(binary_frame, lines);
    vector<Line> d;
    for (int i = 0; i < lines.size(); ++i) {
        d.push_back(Line(Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3])));
    }
    draw_lines(topview_frame, lines);
    //lines = filter_lines(lines);
    //Mat hough_frame = topview_frame.clone();
    double steering_angle = get_steering_angle(lines);
    Line res = approximate(d, steering_angle);

    return res;
}

template<typename T> int sign(T x) {
    if (x == 0) return 0;
    return x < 0 ? -1 : 1;
}

double adjust_angle(Line previous_line, Line current_line) {
    double current_angle = current_line.angle();
    static const int DISTANCE_THRESHOLD = LOCAL_FRAME_WIDTH / 10;
    double difference = current_line.intersect(LOCAL_FRAME_HEIGHT / 2) - previous_line.intersect(LOCAL_FRAME_HEIGHT / 2);

    double multiplier = 2.5;
    double minor = 0.5;
    cerr << "difference = " << difference << endl;
    if (abs(difference) >= DISTANCE_THRESHOLD) {
        cerr << "adjusted\n";
        if (sign(difference) == sign(current_angle)) {
            multiplier += minor;
        } else {
            multiplier -= minor;
        }
    }

    return current_angle * multiplier * (-1);
}
