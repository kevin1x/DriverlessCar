#pragma once

#include "COpticalFlowTracker.h"
#include "CVehicleDetector.h"

using namespace std;

class  CVehicleCounter
	{
	public:
		CVehicleCounter();
		virtual ~CVehicleCounter();

	private:
		CVehicleDetector *vehicleDetector;		
		COpticalFlowTracker *vehicleTracker;		
		std::vector<cv::Point> m_vecBoundingPoint;
		cv::Mat m_image_mask;
		
		int m_iVehicleType;
		int m_iSensitive;
		float m_fScaleFactor;
		int m_iFlag;
		int m_iSizeMin;
		int m_iSizeMax;
				
		int m_iAreaX;
		int m_iAreaY;
		int m_iAreaW;
		int m_iAreaH;
		
		int CreateMaskImage(cv::Mat img_src);
		int CopyVehicleInfo(vector<cv::Rect> rectInfo,vector<_t_Moto> &motoInfo);
		int CheckRectInRect(cv::Rect rect1,cv::Rect rect2);
		int ChangeDetectedCoordinate(vector<cv::Rect> &rectInfo);
		int FilterDetectedVehicle(vector<cv::Rect> &rectInfoIn,int iMinSize,int iMaxSize);
		int FilterMotoInCar(vector<cv::Rect> &rectInfoMoto,vector<cv::Rect> rectInfoCar);
		int SetBoundLineForTracking();
	public:
				
		int SetCountArea(int iX,int iY,int iW,int iH);
		int SetImageMask(cv::Mat imgMask);
		int SetCascadeFile(char* strCascadeVehicle,int iVehicleType);
		int SetParam(int iSensitive,float fScaleFactor,int flag,int iVehicleSizeMin,int iVehicleSizeMax);				

		int Init();
		int CountVehicle(cv::Mat img_resize,std::vector<track> &vecVehicleTracker);

	};
