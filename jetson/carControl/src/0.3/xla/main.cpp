#include <bits/stdc++.h>
#include <opencv/cv.hpp>
#include "xla.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Missing video_file_name argument\n";
        return 1;
    }

    VideoCapture video_capture(argv[1]);
    int number_of_frames = video_capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fps = video_capture.get(CV_CAP_PROP_FPS);
    int width = video_capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = video_capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    printf("Process %d frames, FPS = %d, Size = %dx%d\n", number_of_frames, fps, width, height);

    //Rect ROI = Rect(0, height / 2, width, height / 2);

    VideoWriter bgr_writer;
    bgr_writer.open("bgr.avi", CV_FOURCC('X', 'V', 'I', 'D'), 10, Size(width, height));

    Line last_lane_line;
    bool first_frame = true;
    XLA xla("../topview_auto/params.txt");

    while (true) {
        cerr << "BEGIN #" << video_capture.get(CV_CAP_PROP_POS_FRAMES) << endl;

        static Mat bgr_frame;
        video_capture >> bgr_frame;
        if (bgr_frame.empty()) break;
        int topview_height, topview_width;
        Line lane_line = xla.process_frame(bgr_frame, bgr_writer, topview_height, topview_width);
        double steering_angle = xla.adjust_angle(last_lane_line, lane_line, topview_height, topview_width);
        
        xla.show_angle(bgr_frame, (steering_angle * (-1) / 10 + 90) / 180 * PI);
        imshow("bgr_after", bgr_frame); waitKey(20);
//        bgr_writer << bgr_frame;
        last_lane_line = lane_line;
        
        cerr << "END #" << video_capture.get(CV_CAP_PROP_POS_FRAMES) << endl;
        first_frame = false;
    }

    bgr_writer.release();

    cerr << "Elapsed " << (double)clock() / CLOCKS_PER_SEC << endl;
    return 0;
}
