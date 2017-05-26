
#include "api_lane_detection.h"

enum ConvolutionType {   
/* Return the full convolution, including border */
  CONVOLUTION_FULL, 
  
/* Return only the part that corresponds to the original image */
  CONVOLUTION_SAME,
  
/* Return only the submatrix containing elements that were not influenced by the border */
  CONVOLUTION_VALID
};

void conv2(const cv::Mat &img, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dest) {
	cv::Mat source = img;
	if(CONVOLUTION_FULL == type) {
		source = Mat();
		const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
		copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
	}

	cv::Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
	int borderMode = BORDER_CONSTANT;
	cv::Mat fkernel;
	flip(kernel, fkernel, -1);
	cv::filter2D(source, dest, CV_64F, fkernel, anchor, 0, borderMode);

	if(CONVOLUTION_VALID == type) {
		dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
           .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
	}
}

double sqr(double x) {
	return x * x;
}

bool inRange(double val, double l, double r) {
	return (l <= val && val <= r);
}

void waveletTransform(const cv::Mat& img, cv::Mat& edge, double threshold = 0.15) {
	Mat src = img;
	if (img.channels() == 3)
		cvtColor(img, src, CV_BGR2GRAY);
	double pi = M_PI;
	int SIZE = src.rows;
	int SIZE1 = src.cols;
	double m = 1.0;
	double dlt = pow(2.0, m);
	int N = 20;
	double A = -1 / sqrt(2 * pi);//M_PI = acos(-1.0)
	cv::Mat phi_x = cv::Mat(N, N, CV_64F);
	cv::Mat phi_y = cv::Mat(N, N, CV_64F);
	for(int idx = 1; idx <= N; ++idx)
		for(int idy = 1; idy <= N; ++idy) {
			double x = idx - (N + 1) / 2.0;
			double y = idy - (N + 1) / 2.0;
			double coff = A / sqr(dlt) * exp(-(sqr(x) + sqr(y)) / (2 * sqr(dlt)));
			phi_x.at<double>(idx - 1, idy - 1) = (coff * x);
			phi_y.at<double>(idx - 1, idy - 1) = (coff * y);
		}
	normalize(phi_x, phi_x);
	normalize(phi_y, phi_y);
	cv::Mat Gx, Gy;
	conv2(src, phi_x, CONVOLUTION_SAME, Gx);
	conv2(src, phi_y, CONVOLUTION_SAME, Gy);
	cv::Mat Grads = cv::Mat(src.rows, src.cols, CV_64F);
	for(int i = 0; i < Gx.rows; ++i)
		for(int j = 0; j < Gx.cols; ++j) {
			double x = Gx.at<double>(i, j);
			double y = Gy.at<double>(i, j);
			double sqx = sqr(x);
			double sqy = sqr(y);
			Grads.at<double>(i, j) = sqrt(sqx + sqy);
		}
	double mEPS = 100.0 / (1LL << 52);//matlab eps = 2 ^ -52
	cv::Mat angle_array = cv::Mat::zeros(SIZE, SIZE1, CV_64F);
	for(int i = 0; i < SIZE; ++i)
		for(int j = 0; j < SIZE1; ++j) {
			double p = 90;
			if (fabs(Gx.at<double>(i, j)) > mEPS) {
				p = atan(Gy.at<double>(i, j) / Gx.at<double>(i, j)) * 180 / pi;
				if (p < 0) p += 360;
				if (Gx.at<double>(i, j) < 0 && p > 180)
					p -= 180;
				else if (Gx.at<double>(i, j) < 0 && p < 180)
					p += 180;
			}
			angle_array.at<double>(i, j) = p;
		}
	Mat edge_array = cv::Mat::zeros(SIZE, SIZE1, CV_64F);
	for(int i = 1; i < SIZE - 1; ++i)
		for(int j = 1; j < SIZE1 - 1; ++j) {
			double aval = angle_array.at<double>(i, j);
			double gval = Grads.at<double>(i, j);
			if (inRange(aval,-22.5,22.5) || inRange(aval, 180-22.5,180+22.5)) {
				if (gval > Grads.at<double>(i+1,j) && gval > Grads.at<double>(i-1,j))
					edge_array.at<double>(i, j) = gval;
			}
			else
			if (inRange(aval,90-22.5,90+22.5) || inRange(aval,270-22.5,270+22.5)) {
				if (gval > Grads.at<double>(i, j+1) && gval > Grads.at<double>(i, j-1))
					edge_array.at<double>(i, j) = gval;
			}
			else
			if(inRange(aval,45-22.5,45+22.5) || inRange(aval,225-22.5,225+22.5)) {
				if (gval > Grads.at<double>(i+1,j+1) && gval > Grads.at<double>(i-1,j-1))
					edge_array.at<double>(i,j) = gval;
			}
			else
				if (gval > Grads.at<double>(i+1,j-1) && gval > Grads.at<double>(i-1,j+1))
					edge_array.at<double>(i, j) = gval;
		}
	double MAX_E = edge_array.at<double>(0, 0);
	for(int i = 0; i < edge_array.rows; ++i)
		for(int j = 0; j < edge_array.cols; ++j)
			if (MAX_E < edge_array.at<double>(i, j))
				MAX_E = edge_array.at<double>(i, j);
	edge = Mat::zeros(src.rows, src.cols, CV_8U);
	for(int i = 0; i < edge_array.rows; ++i)
		for(int j = 0; j < edge_array.cols; ++j) {
			edge_array.at<double>(i, j) /= MAX_E;
			if (edge_array.at<double>(i, j) > threshold)
				edge.at<uchar>(i, j) = 255;
			else
				edge.at<uchar>(i, j) = 0;
		}
}

void edgeProcessing(Mat org, Mat &dst, Mat element, string method){
    if (method != "Canny" && method != "Sobel" && method != "Prewitt" && method != "Roberts" && method != "Wavelet") {
        printf("%s is not supported. Please use Canny, Sobel, Prewitt, Roberts or Wavelet\n", method.c_str());
        printf("method is automatically change to Canny\n");
        method = "Canny";
    }
    
    // ------ Canny detection ----------
    if (method == "Canny") {
        int kernel_size = 3;
        int ratio = 5;
        int lowThreshold = 70;
        
        //Canny(org, dst, lowThreshold, lowThreshold*ratio, kernel_size);
        cv::Canny(org, dst,10,100,3);
        cv::dilate(dst, dst, element);
        return;
    }
    
	if (method == "Wavelet") {
        waveletTransform(org, dst);
        //Nang anh
        cv::Mat tmp = cv::getStructuringElement( MORPH_RECT,
                                         Size( 3, 3 ),
                                         Point( 1, 1) );
        cv::dilate(dst, dst, tmp);
		return;
	}
   
    // ------- Sobel, Roberts & Prewitt detection -------
    Mat crossX = (Mat_<double>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, -1); // Sobel
    Mat crossY = (Mat_<double>(3,3) << 1, 2, 1, 0, 0, 0, -1, -2, -1); // Sobel
    if (method == "Prewitt") {
        crossX = (Mat_<double>(3,3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
        crossY = (Mat_<double>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    }
    if (method == "Roberts") {
        crossX = (Mat_<double>(3,3) << 0, 0, 0, 0, 1, 0, 0, 0, -1);
        crossY = (Mat_<double>(3,3) << 0, 0, 0, 0, 0, 1, 0, -1, 0);
    }
    
    int ddepth = -1;
    Point anchor = Point(-1, -1);
    int delta = 0;
    
    Mat dst_v;
    Mat dst_y;
    
    filter2D(org, dst_v, ddepth , crossX, anchor, delta, BORDER_DEFAULT );
    filter2D(org, dst_y, ddepth , crossY, anchor, delta, BORDER_DEFAULT );
    
    addWeighted(dst_v, 0.5, dst_y, 0.5, 0, dst);
    dilate(dst, dst, element );
    
}

int
laneDetect(
        cv::Mat imgGrayOrigin,
        cv::Rect rectDetect,
        vector<vector<cv::Point> > &lineSegments,
        string method
        )
{
    cv::Mat imgCanny;
    vector<cv::Point> aux;
    vector<cv::Vec4i> lines;

    if(imgGrayOrigin.empty())
        return -1;

    cv::Mat imgGray = imgGrayOrigin(rectDetect);

    int erosion_size = 1;
    cv::Mat element = cv::getStructuringElement( MORPH_RECT,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );
    edgeProcessing(imgGray, imgCanny, element, method);

    // Hough
    int houghThreshold = 100;

    cv::HoughLinesP(imgCanny, lines, 2, CV_PI/90, houghThreshold, 10,30);



    // Filter
    for(size_t i=0; i<lines.size(); i++)
    {
        cv::Point pt1, pt2;
        pt1.x = lines[i][0]+rectDetect.x;
        pt1.y = lines[i][1]+rectDetect.y;
        pt2.x = lines[i][2]+rectDetect.x;
        pt2.y = lines[i][3]+rectDetect.y;

        double X = (double)(pt2.x - pt1.x);
        double Y = (double)(pt2.y - pt1.y);

        float  angle = atan2(Y,X)*180/CV_PI;
        float delta = abs(angle);
        int px=(pt1.x+pt2.x)/2;		//// X-Center of line
        int py=(pt1.y+pt2.y)/2;		//// Y-Center of line

        //// Line in the right side
        if( (px>imgGrayOrigin.cols/2) && Y>0 && angle>20 && angle<80)
        //if( (px>imgGrayOrigin.cols/2) && Y>0 && angle>25 && angle<85)
        {
            // Store into vector of pairs of Points for msac
            aux.clear();
            aux.push_back(pt1);
            aux.push_back(pt2);
            lineSegments.push_back(aux);

            //printf("angle = %f\n",angle);
            //cv::line(imgColor,pt1,pt2,cv::Scalar(0,0,255,0),2);
            //cv::imshow("Hough", imgColor);
            //cv::waitKey();
        }
        //// Line in the left side
        if( (px<imgGrayOrigin.cols/2) && Y<0 && angle>-80  && angle<-20 )
        {
            // Store into vector of pairs of Points for msac
            aux.clear();
            aux.push_back(pt1);
            aux.push_back(pt2);
            lineSegments.push_back(aux);

            //printf("angle = %f\n",angle);
            //cv::line(imgColor,pt1,pt2,cv::Scalar(0,0,255,0),2);
            //cv::imshow("Hough", imgColor);
            //cv::waitKey();
        }
    }
    aux.clear();
    lines.clear();
    return 0;
}


void
show_line_segments(
        Mat imgOutput,
        std::vector< std::vector< cv::Point > > &lineSegments,
        std::vector<Mat> &vps)
{
    for( int i = 0; i < lineSegments.size(); i++ )
    {
        vector<cv::Point> line = lineSegments.at(i);
        cv::Point pt1 = line.at(0);
        cv::Point pt2 = line.at(1);

        cv::line(imgOutput,pt1,pt2,cv::Scalar(0,0,255,0),2);
    }

    cv::Point vp(0,0);

    if( vps.size() > 0 )
    {
        double vpNorm = cv::norm(vps[0]);
        if(fabs(vpNorm - 1) < 0.001)
        {
            cout<< endl<< "Norm INFINITE"<< flush;
        }
        vp.x = vps[0].at<float>(0,0);
        vp.y = vps[0].at<float>(1,0);

        cout<< endl<< "VSP: "<< vp.x<< ", "
                             << vp.y;
    }
    else
    {
        cout<< endl<< "No vsp"<< endl<< flush;
    }

    cv::circle(imgOutput, vp, 4, 123, 3);

    // Draw Output
    cv::imshow("Result",imgOutput);

    vps.clear();
    lineSegments.clear();
}


void
api_vanishing_point_init(MSAC &msac)
{

    int mode = MODE_NIETO;
    bool verbose = false;
    cv::Size procSize(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT);

    msac.init(mode, procSize, verbose);

}

void
api_get_vanishing_point(Mat imgGray,
                        Rect roi,
                        MSAC &msac,
                        Point &vp,
                        bool is_show_output,
						string method)
{
    cv::Mat imgPyr;
    std::vector< std::vector< cv::Point > > lineSegments;
    std::vector<Mat> vps;

    // Pyramid
    cv::pyrDown(imgGray, imgPyr, cv::Size(imgGray.cols/2, imgGray.rows/2));
    cv::pyrUp(imgPyr, imgGray, imgGray.size());


    laneDetect(imgGray, roi, lineSegments, method);

    int numVps = 1;

    // Multiple vanishing points

    std::vector<int> numInliers;

    std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;

    // Call msac function for multiple vanishing point estimation

    msac.multipleVPEstimation(lineSegments,
                              lineSegmentsClusters,
                              numInliers,
                              vps,
                              numVps);

    if( vps.size() > 0 )
    {
        vp.x = (int)vps[0].at<float>(0,0);
        vp.y = (int)vps[0].at<float>(1,0);
    }
    else // No vanishing point
    {
        vp.x = 0;
        vp.y = 0;
    }

    if( vp.x < 0 || vp.y < 0 ) { vp.x = 0; vp.y = 0; }

    if( is_show_output )
    {
        Mat imgOutput = imgGray.clone();
        show_line_segments( imgOutput, lineSegments, vps );
    }
    else
    {
        vps.clear();
        lineSegments.clear();
    }

}




void
do_template_matching( Mat img_bin, const Mat &tmp,
                      const string &tmp_name,
                      Point& matchLoc,
                      double &matchVal)
{
    int match_method = CV_TM_CCOEFF_NORMED;

    // Localizing the best match with minMaxLoc
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    /// Create the result matrix
    int result_cols1 = img_bin.cols - tmp.cols + 1;
    int result_rows1 = img_bin.rows - tmp.rows + 1;

    Mat result;
    result.create( result_rows1, result_cols1, CV_32FC1 );

    /// Do the Matching and Normalize
    matchTemplate( img_bin, tmp, result, match_method );

    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    matchLoc = maxLoc;
    matchVal = maxVal;

    matchLoc.x += tmp.cols/2;
    matchLoc.y += tmp.rows/2;

    /// Show me what you got
//    rectangle( img_bin, matchLoc, Point( matchLoc.x + tmp.cols , matchLoc.y + tmp.rows ), Scalar::all(0), 2, 8, 0 );
//    rectangle( result , matchLoc, Point( matchLoc.x + tmp.cols , matchLoc.y + tmp.rows ), Scalar::all(0), 2, 8, 0 );

//    imshow( tmp_name + "_bin", img_bin );
//    imshow( tmp_name + "_res", result );

//    cout<< endl<< tmp_name<< ": MaxVal: "<< maxVal<< flush;
}

void
get_lines(const Mat &imgGray, // input
                         const Rect &roi,    // input
                         const Mat &tmp,
                         const string &tmp_name,
                         vector<Point> &vec_match_loc,
                         vector<double>&vec_match_val)
{

    double matchVal1 = 0.0;
    double matchVal2 = 0.0;
    double matchVal3 = 0.0;
    Point matchLoc1(0,0);
    Point matchLoc2(0,0);
    Point matchLoc3(0,0);
    double thresVal1 = 0.2;
    double thresVal2 = 0.15;
    double thresVal3 = 0.1;

    cv::Rect roi3 = cv::Rect(0, 0,
                            roi.width, roi.height/3);

    cv::Rect roi2 = cv::Rect(roi3.x, roi3.y + roi.height/3,
                            roi.width, roi.height/3);

    cv::Rect roi1 = cv::Rect(roi2.x, roi2.y + roi.height/3,
                            roi.width, roi.height/3);


    // Blur and threshold
    Mat img = imgGray(roi).clone();
    Size ksize(9,9);
    blur(img, img, ksize );

    Mat img_bin;
    threshold( img, img_bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

//    imshow(tmp_name, img_bin);

    Mat img_bin1 = img_bin(roi1).clone();
    string tmp_name1 = tmp_name + "_img1";
    do_template_matching( img_bin1, tmp, tmp_name1, matchLoc1, matchVal1);


    Mat img_bin2 = img_bin(roi2).clone();
    string tmp_name2 = tmp_name + "_img2";
    do_template_matching( img_bin2, tmp, tmp_name2, matchLoc2, matchVal2);


    Mat img_bin3 = img_bin(roi3).clone();
    string tmp_name3 = tmp_name + "_img3";
    do_template_matching( img_bin3, tmp, tmp_name3, matchLoc3, matchVal3);

    bool found1 = true;
    bool found2 = true;
    bool found3 = true;

    if( matchVal1 <= thresVal1 )found1 = false;

    if( matchVal2 <= thresVal2 )found2 = false;

    if( matchVal3 <= thresVal3 )found3 = false;

    if( found1 )
    {
        matchLoc1.x += roi.x + roi1.x;
        matchLoc1.y += roi.y + roi1.y;
        vec_match_loc.push_back(matchLoc1);
        vec_match_val.push_back(matchVal1);
    }

    if( found2 )
    {
        matchLoc2.x += roi.x + roi2.x;
        matchLoc2.y += roi.y + roi2.y;
        vec_match_loc.push_back(matchLoc2);
        vec_match_val.push_back(matchVal2);
    }

    if( found3 )
    {
        matchLoc3.x += roi.x + roi3.x;
        matchLoc3.y += roi.y + roi3.y;
        vec_match_loc.push_back(matchLoc3);
        vec_match_val.push_back(matchVal3);
    }

}

void
api_get_lane_center(Mat &imgGray,
                    Point &center_point,
                    bool is_show_output )
{

    // generate template mask
    Size s(42, 21);
    Mat tmp_left  = Mat::zeros(s, CV_8UC1);
    Mat tmp_right = Mat::zeros(s, CV_8UC1);

    vector<Point> vp_left;

    vp_left.push_back(Point(30,0));
    vp_left.push_back(Point(42,0));
    vp_left.push_back(Point(15,21));
    vp_left.push_back(Point(0,21));

    vector<Point> vp_right;

    vp_right.push_back(Point(0,0));
    vp_right.push_back(Point(12,0));
    vp_right.push_back(Point(42,21));
    vp_right.push_back(Point(27,21));

    fillConvexPoly( tmp_left , Mat(vp_left  ), Scalar( 255, 255, 255 ));
    fillConvexPoly( tmp_right, Mat(vp_right ), Scalar( 255, 255, 255 ));

    //  generate roi
    int frame_width  = imgGray.cols;
    int frame_height = imgGray.rows;

    cv::Rect roi_left = cv::Rect(0, frame_height*3/4,
                            frame_width/2, frame_height/4);

    cv::Rect roi_right = cv::Rect(roi_left.width, roi_left.y,
                            frame_width - roi_left.width, roi_left.height);


    ////////////////////////////////////////////////////////////////////

    vector< Point  > vec_left_points;
    vector< double > vec_left_vals;

    vector< Point  > vec_right_points;
    vector< double > vec_right_vals;

    get_lines( imgGray, roi_left, tmp_left, "left", vec_left_points, vec_left_vals);

    get_lines( imgGray, roi_right, tmp_right, "right", vec_right_points, vec_right_vals);

    int mean_x_left = 0;
    int mean_y_left = 0;

    for( int i = 0; i < vec_left_points.size(); i++ )
    {
        if (is_show_output)
            circle( imgGray, vec_left_points[i], 2, Scalar::all(100), 3);
        mean_x_left += vec_left_points[i].x;
        mean_y_left += vec_left_points[i].y;
    }
    if( vec_left_points.size() > 0)
    {
        mean_x_left /= vec_left_points.size();
        mean_y_left /= vec_left_points.size();
    }


    int mean_x_right = 0;
    int mean_y_right = 0;

    for( int i = 0; i < vec_right_points.size(); i++ )
    {
        if (is_show_output)
            circle( imgGray, vec_right_points[i], 2, Scalar::all(100), 3);
        mean_x_right += vec_right_points[i].x;
        mean_y_right += vec_right_points[i].y;
    }
    if( vec_right_points.size() > 0)
    {
        mean_x_right /= vec_right_points.size();
        mean_y_right /= vec_right_points.size();
    }

//    cout<< endl<< "left size: "<< vec_left_points.size()
//        << "; Mean x left: "<< mean_x_left<< "; y left: "<< mean_y_left<< flush;

    if( mean_x_left > 0 && mean_y_left > 0 && mean_x_right > 0 && mean_y_right > 0)
    {
        center_point.x = (mean_x_left + mean_x_right)/2;
        center_point.y = (mean_y_left + mean_y_right)/2;
    }
    else
    {
        center_point.x = 0;
        center_point.y = 0;
    }

    if( is_show_output )
        circle( imgGray, center_point, 4, Scalar::all(0), 3);

}
