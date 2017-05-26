
#include "api_kinect_cv.h"


int
main( int argc, char** argv )
{

    string video_filename = argv[1];

    cout<< "Opening File: "<< video_filename << endl<< flush;

    cv::VideoCapture video;

    bool isOpenVideo = video.open(video_filename);

    if(!isOpenVideo)
    {
        cout<< "Can not read image" << endl<< flush;
        return 1;
    }

    cv::Mat frame;
    cv::Mat depthMap;

    int frame_width = video.get( CV_CAP_PROP_FRAME_WIDTH );
    int frame_height = video.get( CV_CAP_PROP_FRAME_HEIGHT );

    vector< Rect > rects;
    api_kinect_cv_center_rect_gen( rects, frame_width, frame_height);

    int slice_nb = 3;
    int lower_slice_idx = 3;
    int upper_slice_idx = lower_slice_idx + slice_nb;
    int lower_bound = DIST_MIN + lower_slice_idx * SLICE_DEPTH;
    int upper_bound = lower_bound + slice_nb*SLICE_DEPTH;

    Rect center_rect = rects[lower_slice_idx];
    center_rect = center_rect + Size(0, slice_nb*SLICE_DEPTH);

    Rect intersect;

    char key = 0;

    double st = 0, et = 0, fps = 0;
    double freq = getTickFrequency();

    while( true )
    {

        video.read( frame );

        if(frame.empty())
        {
            cout<< "Can not read image" << endl;
            return 0;
        }
        cv::cvtColor(frame, depthMap, CV_BGR2GRAY);

        Rect roi(0, 132, 640, 238);
        vector< Rect > output_boxes;

        st = getTickCount();

        blur(depthMap(roi), depthMap(roi), Size (3,3));

        api_kinect_cv_get_obtacle_rect( depthMap, output_boxes, roi,
                                        lower_bound, upper_bound );

        et = getTickCount();

        Mat binImg = Mat::zeros(depthMap.size(), CV_8UC1);

//        for( int i = 0; i < rects.size(); i++ )
//            rectangle( binImg, rects[i], Scalar( 255));

//        rectangle( binImg, center_rect, Scalar( 200));

        for( int i = 0; i< output_boxes.size(); i++ )
        {
//            rectangle( binImg, output_boxes[i], Scalar( 255) );

            intersect = output_boxes[i] & center_rect;
            if( intersect.area() != 0 )
            {
                rectangle( binImg, intersect, Scalar( 255) );
                cout<< endl<< "Has Collision detect"<< flush;
            }
        }

        imshow( "BoundingRect", binImg );

        fps = 1.0 / ((et-st)/freq);
        cout<< endl<< "FPS: "<< fps<< flush;

        key = waitKey(0);

        if( key == 27 ) break;
    }

    return 0;
}
