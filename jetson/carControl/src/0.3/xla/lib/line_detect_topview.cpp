#include "line_detect_topview.h"    

// do not set
int morph_elem = 1;
int morph_size = 3;
//

int TV_param = 272;
int low_b = 0, low_g = 0, low_r = 150;
int high_b = 255, high_g = 20, high_r = 255;

double cut_rate = 4.0/5;
int threshold_rate = 150;

Mat Detect_color(Mat frame) {
    vector<Mat> channels;
    Mat threshold, res, hsv, mor, final;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    //split(hsv,channels);
    //imshow("S_channel",channels[1]);
    //imshow("hsv", hsv);
    inRange(hsv, Scalar(low_b, low_g, low_r), Scalar(high_b, high_g, high_r), threshold);
//    cv::imshow("thres", threshold);
    bitwise_and(frame, frame, res, threshold);
    final = res.clone();
    
    cvtColor(final, final, CV_BGR2GRAY);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(80);
    clahe->apply(final, final);
    
    Ptr<CLAHE> clahe1 = createCLAHE();
    clahe1->setClipLimit(10);
    clahe1->apply(final, final);
    
    //mor = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    //morphologyEx( final, final, 1, mor );
    //morphologyEx( final, final, 2, mor );

    //GaussianBlur(final,final,Size(5,5),0,0);
    
    //adaptiveThreshold(final,final,255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,3,3);
	if (final.channels() == 3) cvtColor(final,final,CV_BGR2GRAY);
	cv::threshold(final,final,threshold_rate,255,0);
    return final;
}

Mat Topview_transform(Mat frame){
    std::cerr << "ENTER Topview_transform\n";
    Mat input = frame,output;
    int width = input.cols;
    int height = input.rows;
    Rect ROI(0, int(height * (1-cut_rate)), width, int(height * cut_rate));
    input = input(ROI).clone();
    Point2f inputQuad[4];
    Point2f outputQuad[4];
    Mat lambda( 2, 4, CV_32FC1 );
    lambda = Mat::zeros( input.rows, input.cols, input.type() );

    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input
    inputQuad[0] = Point2f( 0 );
    inputQuad[1] = Point2f( input.cols,0);
    inputQuad[2] = Point2f( input.cols,input.rows);
    inputQuad[3] = Point2f( 0,input.rows );              //
    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = Point2f( 50,0 );
    outputQuad[1] = Point2f( input.cols - 50,0);
    outputQuad[2] = Point2f( input.cols-TV_param,input.rows);
    outputQuad[3] = Point2f( TV_param,input.rows);

    lambda = getPerspectiveTransform( inputQuad, outputQuad );
    // Apply the Perspective Transform just found to the src image
    warpPerspective(input,output,lambda,output.size() );
    //output = Detect_color(output);
    //threshold(output,output,220,255,0);
    std::cerr << "RETURN Topview_transform\n";    
    return output;
}
