#include "CVehicleDetector.h"

using namespace std;

//// Init
CVehicleDetector::CVehicleDetector()
{
	m_strVehicleNormal	= "moto_thang_201309_LBP_6000.xml";
	m_strVehicleBlack = "oto_thang_den_haar.xml";
	
	m_iSizeMin = MOTO_MIN_SIZE;
	m_iSizeMax = m_iSizeMin * MAX_MIN_RATE;
	m_iFlag = 0;
	m_fScaleFactor = WINDOW_UPSCALE_RATE;
}

CVehicleDetector::CVehicleDetector(char* strCascadeVehicle,int iVehicleType,float fScaleFactor,int iSensitive,int iSizeMin,int iSizeMax)
{
	m_strVehicleNormal	= strCascadeVehicle;
	m_strVehicleBlack = "oto_thang_den_haar.xml";
	
	if(iSizeMin>0)	m_iSizeMin = iSizeMin;
	if(iSizeMax>0)	m_iSizeMax = iSizeMax;
	if(iVehicleType > 0)	m_iVehicleType = iVehicleType;
	if(iSensitive > 0)	m_iSensitive= iSensitive;	
	if(fScaleFactor > 0.0001)	m_fScaleFactor = fScaleFactor;
	
}

CVehicleDetector::~CVehicleDetector()
{

}

int CVehicleDetector::SetCascadeFile(char* strCascadeVehicle,int iVehicleType)
{
	if(NULL == strCascadeVehicle)
		return -1;
	m_iVehicleType = iVehicleType;
	m_strVehicleNormal = strCascadeVehicle;

	return 0;
}

int CVehicleDetector::SetParam(int iSensitive,float fScaleFactor,int flag,int iVehicleSizeMin,int iVehicleSizeMax)
{	
	if(iSensitive > 0)
		m_iSensitive = iSensitive;	
	if(flag > 0)
		m_iFlag = flag;
	if(iVehicleSizeMin > 0)
		m_iSizeMin = iVehicleSizeMin;
	else
		m_iSizeMin = MOTO_MIN_SIZE;

	if(iVehicleSizeMax > 0)
		m_iSizeMax = iVehicleSizeMax;
	else
		m_iSizeMax = MAX_MIN_RATE * m_iSizeMin;

	if(fScaleFactor > 0.0001)
		m_fScaleFactor = fScaleFactor;

	return 0;
}

int CVehicleDetector::InitCascade(char* strCascade ,int iVehicleType)
{
	if(NULL != strCascade)
		m_strVehicleNormal = strCascade;

	m_cascadeVehicleNormal =  new  cv::CascadeClassifier( m_strVehicleNormal);
    if(TypeMoto == iVehicleType)
		m_cascadeVehicleBlack = new  cv::CascadeClassifier( m_strVehicleBlack);
	
	if(m_cascadeVehicleNormal->empty())
		return -1;
	
	return 0;
}

int CVehicleDetector::DetectMoto( cv::Mat img_detect,vector<cv::Rect> &rectVehicle)
{
	if(img_detect.empty())
		return -1;
	if(NULL == m_cascadeVehicleNormal)
		return -2;
		
	m_cascadeVehicleNormal->detectMultiScale(img_detect,rectVehicle,m_fScaleFactor,m_iSensitive,m_iFlag ,cv::Size( m_iSizeMin, m_iSizeMin),cv::Size(m_iSizeMax,m_iSizeMax));
	return 0;
}

int CVehicleDetector::DetectCar(  cv::Mat img_detect,  vector<cv::Rect> &rectCar)
{
	if(img_detect.empty())
		return -1;

	if(NULL == m_cascadeVehicleNormal && NULL == m_cascadeVehicleBlack)
		return -2;

	vector<cv::Rect> rectCar_Norm;
	vector<cv::Rect> rectCar_Black;
	m_cascadeVehicleNormal->detectMultiScale(img_detect,rectCar_Norm,m_fScaleFactor,m_iSensitive,m_iFlag ,cv::Size( m_iSizeMin, m_iSizeMin),cv::Size(m_iSizeMax,m_iSizeMax));
	float fScaleMinMax = (float)m_iSizeMax/m_iSizeMin;
	m_cascadeVehicleBlack->detectMultiScale(img_detect,rectCar_Black,m_fScaleFactor,m_iSensitive,m_iFlag ,
											cv::Size(m_iSizeMin - 10,m_iSizeMin - 10), 
											cv::Size( (m_iSizeMin - 10)*fScaleMinMax,(m_iSizeMin - 10)*fScaleMinMax) );

	for(int i=0;i<rectCar_Norm.size();i++)
	{
		cv::Rect rect = rectCar_Norm.at(i);
		rectCar.push_back(rect);
	}
	for(int i=0;i<rectCar_Black.size();i++)
	{
		cv::Rect rect = rectCar_Black.at(i);
		rectCar.push_back(rect);
	}
	return 0;
}		

int CVehicleDetector::DetectVehicle(cv::Mat img_detect, 
									int iVehicleType,
									std::vector<cv::Rect> &rectDetected)
{
    if(TypeMoto == iVehicleType)
		DetectMoto(img_detect,rectDetected);
    if(TypeCar == iVehicleType)
		DetectCar(img_detect,rectDetected);
	return 0;	
}
