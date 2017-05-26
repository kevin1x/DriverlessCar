#pragma once

#include "Param.h"
#include "hungarian.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
//#include <vector>

//namespace DemoVehicleTracking
//{
	#define MAX_CORNERS 200

	typedef struct _BoundingLine
	{
		cv::Point ptStart;
		cv::Point ptEnd;
		int iBoundingType;
	} BoundingLine;

	class COpticalFlowTracker
	{
	public:
		COpticalFlowTracker();
		virtual ~COpticalFlowTracker();

	private:
		int m_objcount;
		int m_cornerCount;

		CvPoint2D32f* cornersA1;
		CvPoint2D32f* cornersB1;

		cv::Mat m_image_cur;
		cv::Mat m_image_pre;
	
		std::vector<BoundingLine> m_lineBounding;

		unsigned int root(unsigned int x);
		int Ndistance(unsigned int x1,unsigned int  x2,unsigned int y1,unsigned int y2);

		int SetImage(cv::Mat img);
		int CreateTracker(int cX,int cY,int r, int vehicleType,std::vector<track> &m_trackers);
		int InitTrackersList(std::vector<_t_Moto> motor_list, int vehicleType,std::vector<track> &m_trackers);
				
		int ResetCorner();
		int RunOpticalFlow(std::vector<track> &m_trackers);
		int PredictAndUpdateTracker1(int vehicleType, std::vector<track> &m_trackers);
		int PredictAndUpdateTracker2(int* row_mate, int size2, int vehicleType,std::vector<track> &m_trackers);

		int UpdateUsingDetectedData(track* trackerInput,int xk,int yk,int rk, int vehicleType,std::vector<track> &m_trackers);
		int UpdateUsingHungarian(std::vector<_t_Moto> motor_list,int vehicleType,std::vector<track> &m_trackers);

		
	public:		
		int GetObjectCount(int vehicleType);
		int SetBoundingLine(cv::Point ptStart, cv::Point ptEnd, int iBoundingType);		
		int hungarian_assign(cv::Mat img_cur,std::vector<_t_Moto> motor_list,cv::Point pt, int iVehicleType,std::vector<track> &m_trackers);		
		int DeleteOldTrackers(cv::Mat img_mask,std::vector<track> &m_trackers);
		int DrawTracker(cv::Mat img, int x0, int y0, int vehicleType,std::vector<track> &m_trackers) ;
	};
//}
