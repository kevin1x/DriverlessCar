//detect object using backproj and hsv
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
Mat src; Mat gray; Mat hsv; Mat hue; Mat sat; Mat val;
vector<Point> limitLineleft,limitLineright;
int bins = 0;

Mat Topview_only(Mat frame);
Mat Hist_and_Backproj(int, void* );

void show_histogram(string const &name, Mat &image , Mat &hist);
int find_max_peak_distribution(Mat hist);
void get_rangeOfpeak(int max_peak, Mat gray_hist, int &down , int &up);

int main( int, char** argv )
{
	const char* window_image = "Source image";
	namedWindow( window_image, WINDOW_AUTOSIZE );
	VideoCapture cap;
	cap.open(argv[1]);
	int count_frame = 1; 
	
	while (1)	
	{
		cap >> src;
		if (src.empty()) break;
		src = Topview_only(src);
		cvtColor( src, gray, COLOR_BGR2GRAY );
		
		imshow( "gray", gray );		

		//filter(gray);
		
		//********get line limited of topview image********
		if(count_frame == 1)
		{			
			for(int i = 0;i < gray.rows; ++i)
			{
				for(int j = 0;j < gray.cols; ++j)
				{
					Scalar ref = gray.at<uchar>(i,j);				
					int  intensity = ref.val[0];
					if (intensity != 0) 
					{
						limitLineleft.push_back(Point(j,i));
						break;
					}
				}

				for(int j = gray.cols - 1;j >= 0; --j)
				{
					Scalar ref = gray.at<uchar>(i,j);				
					int  intensity = ref.val[0];
					if (intensity != 0) 
					{
						limitLineright.push_back(Point(j,i));
						break;
					}
				}
			}
			++count_frame;
		}


		//********detect object by hist gray********
		Mat gray_hist = gray.clone();					// create histogram of gray image
		//medianBlur ( gray_hist, gray_hist, 21 );
		GaussianBlur( gray_hist, gray_hist, Size( 15, 15 ), 0, 0 );
		imshow( "grayBlur", gray_hist );		

		Mat hist;      									// create matrix for histogram
		show_histogram("gray_hist", gray_hist , hist);	// calculate historgram and show	
		
		int max_peak = find_max_peak_distribution(hist);//find max peak distribution of historgram
		cout<<"max peak : "<<max_peak<<endl;

		int down,up;
		get_rangeOfpeak(max_peak,gray_hist,down,up);		


		Mat binImg = gray.clone();
		for(int i = 0;i < gray.rows; ++i)
		for(int j = 0;j < gray.cols; ++j) binImg.at<uchar>(i,j) = 0;
	
		for(int i = 0;i < gray.rows; ++i)
		for(int j = 0;j < gray.cols; ++j)
		{	
			Scalar ref = gray.at<uchar>(i,j);				
			int  intensity = ref.val[0];
			if ((intensity > 170)||((down <= intensity)&&(intensity <= up))) binImg.at<uchar>(i,j) = 255;
			//if ((down <= intensity)&&(intensity <= up)) binImg.at<uchar>(i,j) = 255;
		}


		//get bin have low frequency
		int bin_low_freq[256];					// bin have height <5 (255-5 = 250) in gray_hist is consider low frequency
		int index = 0;
		for (int j = 1;j < gray_hist.cols; j++)		
		{	
			Scalar ref = gray_hist.at<uchar>(253,j);				
			int intensity = ref.val[0];
			if (intensity == 0) 
			{
				bin_low_freq[index] = j;
				++index;
			}
		}
 
		//make while all pixel have bin have low frequency
		for(int i = 0;i < gray.rows; ++i)
		for(int j = 0;j < gray.cols; ++j)
		{
			Scalar ref = gray.at<uchar>(i,j);				
			int  intensity = ref.val[0];	
			for (int k = 0; k < index; k++)
			if (intensity == bin_low_freq[k]) binImg.at<uchar>(i,j) = 255;
		}

		imshow("bin",binImg);




		//********detect object by backprojection********
		cvtColor( src, hsv, COLOR_BGR2HSV );
		imshow( "hsv", hsv );		
		hue.create( hsv.size(), hsv.depth() );
		sat.create( hsv.size(), hsv.depth() );
		val.create( hsv.size(), hsv.depth() );
		int ch[] = { 0,0 , 1,0 ,2,0};
		mixChannels( &hsv, 1, &hue, 1, ch, 1 );
		//mixChannels( &hsv, 1, &sat, 2, ch, 2 );
		//mixChannels( &hsv, 1, &val, 3, ch, 3 );
		imshow( "hue", hue );
		//filter(hue);
		
		//imshow( "huefilter", hue );
		
		Mat backprojImg = Hist_and_Backproj(0, 0);		

		
		//make all lane to white in backprojImg
		for(int i = 0;i < backprojImg.rows; ++i)
		for(int j = limitLineleft[i].x - 2;j < limitLineright[i].x + 2; ++j)
		{	
			Scalar ref = gray.at<uchar>(i,j);				
			int  intensity = ref.val[0];
			if (intensity > 200) backprojImg.at<uchar>(i,j) = 255;
		}
		
		imshow( "BackProj", backprojImg );


		//********Preprocess********
		Mat binImgdst,backprojImgdst;
		int morph_size = 2;
	    Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ));
		morphologyEx( binImg, binImgdst, MORPH_CLOSE, element, Point(-1,-1), 1 );   
		morphologyEx( backprojImg, backprojImgdst, MORPH_CLOSE, element, Point(-1,-1), 1);   

		imshow( "binImgdst", binImgdst );
		imshow( "backprojImgdst", backprojImgdst );

		//********Create And image and Or image********
		Mat OrImg = gray.clone();
		Mat AndImg = gray.clone();
		for(int i = 0;i < gray.rows; ++i)
		for(int j = 0;j < gray.cols; ++j) 
		{
			OrImg.at<uchar>(i,j) = 255;
			AndImg.at<uchar>(i,j) = 255;
		}

		for(int i = 0;i < gray.rows; ++i)
		for(int j = 0;j < gray.cols; ++j)
		{
			Scalar refBin = binImgdst.at<uchar>(i,j);				
			int  intensityBin = refBin.val[0];
			Scalar refBack = backprojImgdst.at<uchar>(i,j);				
			int  intensityBack = refBack.val[0];

			if (intensityBin == intensityBack) AndImg.at<uchar>(i,j) = intensityBack;									
			if ((intensityBin == 0)||(intensityBack == 0)) OrImg.at<uchar>(i,j) = 0;										
		}


		imshow( "And", AndImg );
		imshow( "Or", OrImg );


		//********contours********
		Mat img = OrImg.clone();
		//change unuse image part limit to white for find contours
		for(int i = 0;i < img.rows; ++i)	
		{
			for(int j = 0;j < limitLineleft[i].x+2; ++j) 	img.at<uchar>(i,j) = 255;
			for(int j = limitLineright[i].x-2;j < img.cols ; ++j) img.at<uchar>(i,j) = 255;
		}
		imshow("img",img);	

		//find contours
		Mat drawing = Mat::zeros( img.size(), CV_8U );
		vector<vector<Point> > contours;
		findContours(img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);          	// find all contours

		if (contours.size()>0)
		{
			for (unsigned int i = 0; i < contours.size(); i++)                 			// for each contour
				drawContours(drawing, contours, i, Scalar(255,255,255));				
			imshow("contours",drawing);	
		}

	

		imshow( window_image, src );
		waitKey(0);
	}
	return 0;
}


Mat Topview_only(Mat frame){
	int TV_param = 230;
	double cut_rate = 3.0/5;
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
	return output;
}



//return backproject image
Mat Hist_and_Backproj(int, void* )
{
	MatND hist;
	int histSize = MAX( bins, 2 );
	float hue_range[] = { 0, 90 };
	const float* ranges = { hue_range };
	calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
	normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );
	MatND backproj;
	calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

	//change unuse image part limit to black 
	for(int i = 0;i < backproj.rows; ++i)	
	{
		for(int j = 0;j < limitLineleft[i].x; ++j) 	backproj.at<uchar>(i,j) = 0;
		for(int j = limitLineright[i].x;j < backproj.cols ; ++j) backproj.at<uchar>(i,j) = 0;
	}
	
	//change background to white , because almost image is background-> get the largest part to white and the remain (object) to black
	int count0 = 0,count255 = 0;
	for(int i = 0;i < backproj.rows; ++i)	
	for(int j = limitLineleft[i].x;j < limitLineright[i].x; ++j)
	{
		Scalar ref = backproj.at<uchar>(i,j);				
		int  intensity = ref.val[0];
		if (intensity == 0) ++count0;
		else if (intensity == 255) ++count255;
	}
	
	if (count255<count0)
	{
		for(int i = 0;i < backproj.rows; ++i)
		for(int j = limitLineleft[i].x;j < limitLineright[i].x; ++j)
		{
			Scalar ref = backproj.at<uchar>(i,j);				
			int  intensity = ref.val[0];
			if (intensity == 0) backproj.at<uchar>(i,j) = 255;
			else if (intensity == 255) backproj.at<uchar>(i,j) = 0;
		}	
	}
	
	return backproj;
}


void show_histogram(string const &name, Mat &image , Mat &hist)
{
    // Range bin of histogram of gray
    int bins = 256; 
    int histSize[] = {bins};
    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};

    int channels[] = {0};

    // create matrix for histogram visualization
    int const hist_height = 256;

	Size s(hist_height, bins);
	Mat hist_image = Mat::zeros( s, CV_8U );

    calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

    //double max_val=0;
    //minMaxLoc(hist, 0, &max_val);

    // find max value of bin
	int max_bin_val = 0;
	int bin_max_val;
    for(int i = 1; i < bins; i++) 
	{
        float const binVal = hist.at<float>(i);        
		//cout<<i<<" : "<<binVal<<endl;
		if (binVal > max_bin_val) 
		{
			max_bin_val = binVal;
			bin_max_val = i;
		}
    }
	//cout<<"Max value : "<<max_bin_val<<" at bin "<<bin_max_val<<endl;


	//draw historgram
    for(int i = 1; i < bins; i++) 
	{
        float const binVal = hist.at<float>(i);
        int   const height = cvRound(binVal*hist_height/max_bin_val);
        line
            ( hist_image
            , Point(i, hist_height-height), Point(i, hist_height)
            , Scalar::all(255)
            );
	}

    imshow(name, hist_image);
	image = hist_image.clone();
}

int find_max_peak_distribution(Mat hist)
{
	int bins = 256;	//range bin of histogram of gray
	int offset = 30;
	// find max value of bin 
	int max_bin_val = 0;
	int bin_max_val;
    for(int i = 1; i < bins; i++) 
	{
        float const binVal = hist.at<float>(i);        
		if (binVal > max_bin_val) 
		{
			max_bin_val = binVal;
			bin_max_val = i;
		}
    }

	//find peaks at downer and upper of the highest peak
	int down_limit = bin_max_val - offset;
	int max_bin_val_down = 0;
	int bin_max_val_down;
	if (down_limit > 20)
	{		
		for(int i = 1; i < down_limit; i++) 
		{
			float const binVal = hist.at<float>(i);        
			if (binVal > max_bin_val_down) 
			{
				max_bin_val_down = binVal;
				bin_max_val_down = i;
			}
		}
		//cout<<"Max value : "<<max_bin_val_down<<" at bin "<<bin_max_val_down<<endl;
	}

	int up_limit = bin_max_val + offset;
	int max_bin_val_up = 0;
	int bin_max_val_up;
	if (up_limit < bins - 20)
	{	
		for(int i = up_limit+1; i < bins; i++) 
		{
			float const binVal = hist.at<float>(i);        
			if (binVal > max_bin_val_up) 
			{
				max_bin_val_up = binVal;
				bin_max_val_up = i;
			}
		}
		//cout<<"Max value : "<<max_bin_val_up<<" at bin "<<bin_max_val_up<<endl;
	}

	if (abs(bin_max_val - bin_max_val_down) < 20) max_bin_val_down = 0;
	if (abs(bin_max_val - bin_max_val_up) < 20) max_bin_val_up = 0;

	//get distribution of peak have largest covariance
	double avg_sum_down = 0;	
	if (max_bin_val_down > 0) 		//exist peak in down side of highest peak
	{		
		int limit;
		if (bin_max_val_down - offset < 1) limit = 1;
		else limit = bin_max_val_down - offset;
		for (int i = limit;i<=bin_max_val_down + offset;i++)	
		{
			float const binVal = hist.at<float>(i);        
			avg_sum_down += binVal;		
		}
		avg_sum_down = avg_sum_down / ((bin_max_val_down + offset) - limit + 1);
		//cout<<"range down : "<<((bin_max_val_down + offset) - limit + 1)<<endl;
	}

	double avg_sum_up = 0;	
	if (max_bin_val_up > 0)		//exist peak in up side of highest peak
	{		
		int limit;
		if (bin_max_val_up + offset >= bins) limit = bins - 1;
		else limit = bin_max_val_up + offset;
		for (int i = bin_max_val_up - offset; i <= limit; i++)	
		{
			float const binVal = hist.at<float>(i);        
			avg_sum_up += binVal;		
		}
		avg_sum_up = avg_sum_up / (limit - (bin_max_val_up - offset) + 1);
		//cout<<"range up : "<<(limit - (bin_max_val_up - offset) + 1)<<endl;
	}


	double avg_sum = 0;	
	if (down_limit < 1) down_limit = 1;
	if (up_limit > bins - 1) up_limit = bins - 1;
	
	for (int i = down_limit; i <= up_limit; i++)	
	{
		float const binVal = hist.at<float>(i);        
		avg_sum += binVal;		
	}
	avg_sum = avg_sum / (up_limit - down_limit + 1);
	//cout<<"range : "<<(up_limit - down_limit + 1)<<endl;

	//check max covariance distribution
	int max = 0;
	int check;		
	if (avg_sum_down > max) 
	{
		max = avg_sum_down;
		check = 1;
	}
	if (avg_sum > max) 
	{
		max = avg_sum;
		check = 2;
	}
	if (avg_sum_up > max) 
	{
		max = avg_sum_up;
		check = 3;
	}

	//cout<<"avg down : "<<avg_sum_down<<endl;
	//cout<<"avg : "<<avg_sum<<endl;
	//cout<<"avg up : "<<avg_sum_up<<endl;

	if (check == 1) return bin_max_val_down;
	if (check == 2) return bin_max_val;
	if (check == 3) return bin_max_val_up;

}



void get_rangeOfpeak(int max_peak, Mat gray_hist , int &down , int &up)
{
	int offsetbin = 5;					//offset to check bin in range
	int bins = 256; 					//range bin of histogram of gray
	int thresh = 230;					//threshhold of heigh of bin in range of distribution of max peak

	up = -1;
	down = -1;
	//find range down
	for (int i = max_peak;i > 0 ;i--)					
	{
		int limit;
		if (i - offsetbin < 1) limit = 1;
		else limit = i - offsetbin;
		bool inRange = false;		//check bin if in range of distribution of max peak
		for (int j = i;j >= limit;j--) 		
		{
			//gray_hist 's size : 256x256
			Scalar ref = gray_hist.at<uchar>(thresh,j);				
			int  intensity = ref.val[0];
			if (intensity != 0)	
			{
				inRange = true;
				break;
			}						
		}
		if (!inRange) 
		{
			down = i+1;
			break;
		}
	}

	//find range up
	for (int i = max_peak;i < bins ;i++)					
	{
		int limit;
		if (i + offsetbin > bins - 1) limit = bins - 1;
		else limit = i + offsetbin;
		bool inRange = false;		//check bin if in range of distribution of max peak
		for (int j = i;j <= limit;j++) 		
		{
			//gray_hist 's size : 256x256
			Scalar ref = gray_hist.at<uchar>(thresh,j);				
			int  intensity = ref.val[0];
			if (intensity != 0)	
			{
				inRange = true;
				break;
			}						
		}
		if (!inRange) 		
		{
			up = i-1;
			break;
		}
	}
	if (down == -1) down = 1;
	if (up == -1) up = 255;


	circle( gray_hist,Point(down,thresh),3,Scalar( 0, 0, 0 ),2,8 );
	circle( gray_hist,Point(up,thresh),3,Scalar( 0, 0, 0 ),2,8 );
	imshow("range hist",gray_hist);

	cout<<"down :"<<down<<endl;
	cout<<"up :"<<up<<endl;
}
