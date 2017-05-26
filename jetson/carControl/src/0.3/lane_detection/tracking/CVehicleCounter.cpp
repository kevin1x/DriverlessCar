#include "CVehicleCounter.h"
#include "time.h"

using namespace std;
//using namespace DemoVehicleTracking;

CVehicleCounter::CVehicleCounter()
{

	m_iAreaX = 0;
	m_iAreaY = 0;
	m_iAreaW = 0;
	m_iAreaH = 0;
	
	m_iSizeMin = MOTO_MIN_SIZE;
	m_iSizeMax = MAX_MIN_RATE*m_iSizeMin;
	m_fScaleFactor= WINDOW_UPSCALE_RATE;
	m_iFlag = 0;
	vehicleDetector = new CVehicleDetector();	
	vehicleTracker = new COpticalFlowTracker();

	//vehicleDetector->SetParam(m_iSensitive,m_fScaleFactor,m_iFlag,m_iSizeMin,m_iSizeMax);
}
CVehicleCounter::~CVehicleCounter()
{
}

int CVehicleCounter::SetCascadeFile(char* strCascadeVehicle,int iVehicleType)
{
	m_iVehicleType = iVehicleType;
	int rs = vehicleDetector->SetCascadeFile(strCascadeVehicle,iVehicleType);	
	return rs;
}

int CVehicleCounter::SetParam(int iSensitive,float fScaleFactor,int flag,int iVehicleSizeMin,int iVehicleSizeMax)
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

	int rs = vehicleDetector->SetParam(iSensitive,fScaleFactor,flag,iVehicleSizeMin,iVehicleSizeMax);
	return rs;
}

int CVehicleCounter::SetCountArea(int iX,int iY,int iW,int iH)
{
	if(iX > 0)	m_iAreaX = iX;
	if(iY > 0)	m_iAreaY = iY;
	if(iW > 0)	m_iAreaW = iW;
	if(iH > 0)	m_iAreaH = iH;
	
	return 0;
}

int CVehicleCounter::SetBoundLineForTracking()
{
	cv::Point ptTopLeft = cv::Point(m_iAreaX,m_iAreaY);
	cv::Point ptTopRight = cv::Point(m_iAreaX + m_iAreaW,m_iAreaY);
	cv::Point ptBotRight = cv::Point(m_iAreaX + m_iAreaW,m_iAreaY + m_iAreaH);
	cv::Point ptBotLeft = cv::Point(m_iAreaX,m_iAreaY + m_iAreaH);

    vehicleTracker->SetBoundingLine(ptTopLeft,ptTopRight,TypeTop);
    vehicleTracker->SetBoundingLine(ptTopRight,ptBotRight,TypeRight);
    vehicleTracker->SetBoundingLine(ptBotRight,ptBotLeft,TypeBottom);
    vehicleTracker->SetBoundingLine(ptBotLeft,ptTopLeft,TypeLeft);

	m_vecBoundingPoint.push_back(ptTopLeft);
	m_vecBoundingPoint.push_back(ptTopRight);
	m_vecBoundingPoint.push_back(ptBotRight);
	m_vecBoundingPoint.push_back(ptBotLeft);


	//cv::Point ptTopLeft = cv::Point(134,233);
	//cv::Point ptTopRight = cv::Point(580,60);
	//cv::Point ptBotRight = cv::Point(1050,200);
	//cv::Point ptBotLeft = cv::Point(525,670);
    //vehicleTracker->SetBoundingLine(ptTopLeft,ptTopRight,TypeTop);
    //vehicleTracker->SetBoundingLine(ptTopRight,ptBotRight,TypeRight);
    //vehicleTracker->SetBoundingLine(ptBotRight,ptBotLeft,TypeBottom);
    //vehicleTracker->SetBoundingLine(ptBotLeft,ptTopLeft,TypeLeft);	//
    //trackerCar->SetBoundingLine(ptTopLeft,ptTopRight,TypeTop);
    //trackerCar->SetBoundingLine(ptTopRight,ptBotRight,TypeRight);
    //trackerCar->SetBoundingLine(ptBotRight,ptBotLeft,TypeBottom);
    //trackerCar->SetBoundingLine(ptBotLeft,ptTopLeft,TypeLeft);
	//m_vecBoundingPoint.push_back(ptTopLeft);
	//m_vecBoundingPoint.push_back(ptTopRight);
	//m_vecBoundingPoint.push_back(ptBotRight);
	//m_vecBoundingPoint.push_back(ptBotLeft);
	//cv::Rect rectBounding = cv::boundingRect(m_vecBoundingPoint);
	//m_iAreaX = rectBounding.x;
	//m_iAreaY = rectBounding.y;
	//m_iAreaW = rectBounding.width;
	//m_iAreaH = rectBounding.height;

	return 0;
}

int CVehicleCounter::SetImageMask(cv::Mat imgMask)
{
	if(imgMask.empty())
		return -1;
	m_image_mask = imgMask.clone();
	return 0;
}

int CVehicleCounter::CreateMaskImage(cv::Mat img_src)
{
	if(img_src.size().width < 1 || img_src.size().height< 1)
		return -1;

	m_image_mask = cv::Mat::zeros(img_src.size(),CV_8U);
	//// Set bounding line for detect and tracking
	//// TO DO: pubic this function
	SetBoundLineForTracking();

	cv::fillConvexPoly(m_image_mask,m_vecBoundingPoint,cv::Scalar(255,255,255,255));
	return 0;
}

int CVehicleCounter::ChangeDetectedCoordinate(vector<cv::Rect> &rectInfo)
{
	if(rectInfo.size() < 1)
		return 0;
	for(unsigned int i=0;i< rectInfo.size();i++)
	{
		rectInfo.at(i).x = rectInfo.at(i).x + m_iAreaX;
		rectInfo.at(i).y = rectInfo.at(i).y + m_iAreaY;
	}
	return 0;
}
int CVehicleCounter::CopyVehicleInfo(vector<cv::Rect> rectInfo,vector<_t_Moto> &motoInfo)
{
	int numVehicle = 0;
	if(rectInfo.size()<1)
		return 0;
	
	for(unsigned int i=0;i< rectInfo.size();i++)
	{
		cv::Rect rect = rectInfo.at(i);
		_t_Moto tMoto;
		tMoto.vehicleCenterX = rect.x + rect.width/2;
		tMoto.vehicleCenterY = rect.y + rect.height/2;
		tMoto.vehicleRadius = rect.width/2;
		tMoto.vehicleSizeW =  rect.width;
		tMoto.vehicleSizeH = rect.height;
		motoInfo.push_back(tMoto);
		numVehicle++;		
	}
	return numVehicle;
}

int CVehicleCounter::CheckRectInRect(cv::Rect rect1,cv::Rect rect2)
{
	if(rect1.contains(cv::Point(rect2.x + 1,rect2.y + 1)) || rect1.contains(cv::Point(rect2.x + rect2.width - 2,rect2.y + rect2.height - 2)))
		return 1;	//// rect2 is in rect1
	if(rect2.contains(cv::Point(rect1.x + 1,rect1.y + 1)) || rect2.contains(cv::Point(rect1.x + rect1.width - 2,rect1.y + rect1.height - 2)))
		return -1;	//// rect1 is in rect2

	//// Rect overload
	//int centerX1 = rect1.x + rect1.width/2;
	//int centerX2 = rect2.x + rect2.width/2;
	//int centerY1 = rect1.y + rect1.height/2;
	//int centerY2 = rect1.y + rect2.height/2;
	//double distance = (centerX1-centerX2)*(centerX1-centerX2) + (centerY1-centerY2)*(centerY1-centerY2);
	//if(distance < rect1.width/2)
	//	return -1;
	//return 0;
}

int CVehicleCounter::FilterDetectedVehicle(vector<cv::Rect> &rectInfoIn,int iMinSize,int iMaxSize)
{
	if(rectInfoIn.size() < 1)
		return 0;

	vector<cv::Rect> rectInfo1;
	//// Reject big vehicle and vehicle outside of interesting area
	for(unsigned int i=0;i<rectInfoIn.size();i++)
	{	
		if( /*(rectInfoIn.at(i).width > iMinSize) && */
			rectInfoIn.at(i).width < iMaxSize  && 
			m_image_mask.at<unsigned char>(rectInfoIn.at(i).y + rectInfoIn.at(i).height/2,rectInfoIn.at(i).x + rectInfoIn.at(i).width/2) > 0)
			rectInfo1.push_back(rectInfoIn.at(i));
	}
	rectInfoIn.clear();

	//// Reject overlap vehicle
	vector<cv::Rect> rectInfo2;
	for(unsigned int i=0;i<rectInfo1.size();i++)
		rectInfo2.push_back(rectInfo1.at(i));
		
	for(unsigned int i=0;i<rectInfo1.size();i++)
	{
		int rs = 0;
		for(unsigned int j = 0;j<rectInfo2.size();j++)
		{
			rs += CheckRectInRect(rectInfo1.at(i),rectInfo2.at(j));
		}
		if(rs > -1)
			rectInfoIn.push_back(rectInfo1.at(i));
	}
	return 0;
}

int CVehicleCounter::FilterMotoInCar(vector<cv::Rect> &rectInfoMoto,vector<cv::Rect> rectInfoCar)
{
	vector<cv::Rect> rectInfo;
	for(unsigned int i=0;i<rectInfoMoto.size();i++)
		rectInfo.push_back(rectInfoMoto.at(i));

	rectInfoMoto.clear();

	for(unsigned int i=0;i<rectInfo.size();i++)
	{
		int rs = 0;
		for(unsigned int j = 0;j<rectInfoCar.size();j++)
		{
			rs = CheckRectInRect(rectInfoCar.at(i),rectInfo.at(j));
			if(rs != 0)
				break;
		}
		if(rs != 0)
			continue;
		rectInfoMoto.push_back(rectInfo.at(i));
	}

	return 0;
}

int CVehicleCounter::Init()
{		
	//// Load Cascade	
	int rsLoadCascade = vehicleDetector->InitCascade();	
	if(0 != rsLoadCascade)
		return -1;
	return 0;
}

int CVehicleCounter::CountVehicle(cv::Mat img_intput,std::vector<track> &vecVehicleTracker)
{
	if(img_intput.empty())	
		return -1;

	//// Create mask image
	if(m_image_mask.empty())	
		int rs = CreateMaskImage(img_intput);

	//// Set image area
	cv::Rect rectArea(m_iAreaX,m_iAreaY,m_iAreaW,m_iAreaH);
	if(m_iAreaX + m_iAreaW > img_intput.cols - 1)
		rectArea.width = img_intput.cols - 1 - m_iAreaX;
	if(m_iAreaY + m_iAreaH > img_intput.rows - 1)
		rectArea.height = img_intput.rows - 1 - m_iAreaY;

	cv::Mat img_detect = img_intput(rectArea);
	vector<cv::Rect> rectVehicle;	
	//// Detect in cropped area
	vehicleDetector->DetectVehicle(img_detect,m_iVehicleType,rectVehicle);
	//// Change coordinate according to full image
	ChangeDetectedCoordinate(rectVehicle);
	//// Filter overlap vehicle
	FilterDetectedVehicle(rectVehicle,(double)m_iSizeMin/1.4 - 1, m_iSizeMax * 1.5);
	
	////// Copy Info after detect, convert coordinate to full frame
	vector<_t_Moto> infoMoto;	
	int numMoto = CopyVehicleInfo(rectVehicle,infoMoto);	

	//// Moto tracking	
	vehicleTracker->hungarian_assign(img_intput,infoMoto,cv::Point(0,0),m_iVehicleType,vecVehicleTracker);
	vehicleTracker->DeleteOldTrackers(m_image_mask,vecVehicleTracker);

	////// Draw result
	//vehicleTracker->DrawTracker(img_intput,0,0, m_iVehicleType,vecVehicleTracker);
	
	return 0;

}
