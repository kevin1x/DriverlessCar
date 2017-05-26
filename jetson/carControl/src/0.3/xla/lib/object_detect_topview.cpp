#include "line_detect_topview.h"

int offset = 20;
const int offset_frame = 50;

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
    std::cerr << "RETURN Topview_only\n";
	return output;
}


Mat getObjectMat(Mat frame)
{
	int count_fr = 0;
	int bg,bg_ref;
	Mat bin;
	int count_pixel = 0;
	bool noise = true;
	int sum,count;	


	++count_fr;		    
	
	//convert to topview
	frame = Topview_only(frame);

	cvtColor(frame, frame, CV_BGR2GRAY);		
	medianBlur(frame,frame,7);

	if (count_fr == 1)
	{					
		sum = 0 , count = 0;
		for (int i=frame.rows - 10 ; i<frame.rows ; i++)		
		for (int j=frame.cols/2 - 30 ; j<frame.cols/2 + 30 ; j++)		
		{			
			Scalar ref = frame.at<uchar>(i,j);				
			int  intensity = ref.val[0];
			sum += intensity;
			count += 1;
		}
		bg = sum / count;
		cout<<"bg = "<<bg<<endl;	
		bg_ref = bg;		
	}

	bg = bg_ref;
	noise = true;
	offset = 20;


	while (noise)
	{				
		bin = Mat::zeros(frame.rows, frame.cols, frame.type());
		
		int ii =  frame.rows-1;
		bool change_bg = false;

		for (int i = frame.rows-1 ; i>0 ;i--)
		{
			if (ii-i >= 50) 
			{
				ii = i;
				change_bg = true;
			}

			sum = 0;
			count = 0;
			for (int j = 0 ; j< frame.cols;j++)
			{
				Scalar ref = frame.at<uchar>(i,j);				
				int  intensity = ref.val[0];
				if ((intensity < 200)&&((intensity < bg - offset)||(intensity > bg + offset))) bin.at<uchar>(i,j) = 255;
				else if ((intensity > bg - offset/3)&&(intensity < bg + offset/3)) 
				{
					sum += intensity;
					++count;
				} 
			}
			
			if (change_bg) 
			{
				if (count > 100000)bg = sum / count;
				change_bg = false;
			}
		}

		noise = false;
		count_pixel = 0;
		for (int i = bin.rows*3/4 ; i< bin.rows;i++)
		{
			for (int j = 0 ; j< bin.cols;j++) 
			{
				Scalar ref = bin.at<uchar>(i,j);				
				int  intensity = ref.val[0];
				if (intensity > 0) 
				{
					++count_pixel;
					noise = true;							
				}
			}
		}

		if (count_pixel < 200) noise = false;
		else offset += 5;
		if (offset > 100) break;
		cout<<"offset = "<<offset<<endl;
	}


	bin = Mat::zeros(frame.rows, frame.cols, frame.type());
	int ii =  frame.rows-1;
	bool change_bg = false;

	for (int i = frame.rows-1 ; i>0 ;i--)
	{
		if (ii-i >= 50) 
		{
			ii = i;
			change_bg = true;
		}

		sum = 0;
		count = 0;
		for (int j = 0 ; j< frame.cols;j++)
		{
			Scalar ref = frame.at<uchar>(i,j);				
			int  intensity = ref.val[0];
			if ((intensity < 200)&&((intensity < bg - offset)||(intensity > bg + offset))) bin.at<uchar>(i,j) = 255;
			else if ((intensity > bg - offset/2)&&(intensity < bg + offset/2)) 
			{
				sum += intensity;
				++count;
			} 
		}
		if ((change_bg)&&(count > 300))
		{
			if ((0 < sum / count)&&(sum / count < 255)) bg = sum / count;					
			change_bg = false;
		}
	}


	int morph_size = 4;
	Mat element = getStructuringElement( 2, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx( bin, bin, 2, element );

	return bin;
}
