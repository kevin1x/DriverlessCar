#pragma once

#include "Param.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <stdio.h>

	class CVehicleDetector
	{
	public:
		CVehicleDetector();
		CVehicleDetector(char* strCascadeVehicle,int iVehicleType,float fScaleFactor,int iSensitive,int iSizeMin,int iSizeMax);
		virtual ~CVehicleDetector();
	private:
	
		int m_iVehicleType;
		char *m_strVehicleNormal;
		char *m_strVehicleBlack;
								
		int m_iSensitive;
		int m_iFlag;
		int m_iSizeMin;
		int m_iSizeMax;
		float m_fScaleFactor;
		
		cv::CascadeClassifier *m_cascadeVehicleNormal;
		cv::CascadeClassifier *m_cascadeVehicleBlack;
		int DetectMoto( cv::Mat img_detect, std::vector<cv::Rect> &rectVehicle);
		int DetectCar( cv::Mat img_detect, std::vector<cv::Rect> &rectCar);

	public:
		
		int SetCascadeFile(char* strCascadeVehicle,int iVehicleType);		
        int InitCascade(char* strCascade = NULL,int iVehicleType = TypeMoto);
		int SetParam(int iSensitive,float fScaleFactor,int flag,int iVehicleSizeMin,int iVehicleSizeMax);
		
		int DetectVehicle(	cv::Mat img_detect, 
							int iVehicleType,
							std::vector<cv::Rect> &rectDetected);
	};
