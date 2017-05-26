#include "COpticalFlowTracker.h"


using namespace std;
//using namespace DemoVehicleTracking;

////>> Miscellaneous functionsMiscellaneous
//fast sqrt
unsigned int COpticalFlowTracker::root(unsigned int x){
    unsigned int a,b;
    b     = x;
    a = x = 0x3f;
    x     = b/x;
    a = x = (x+a)>>1;
    x     = b/x;
    a = x = (x+a)>>1;
    x     = b/x;
    x     = (x+a)>>1;
    return(x);  
}

int COpticalFlowTracker::Ndistance(unsigned int x1,unsigned int  x2,unsigned int y1,unsigned int y2)
{
	int dx=x1-x2;
	int dy=y1-y2;
	return root(dx*dx+dy*dy);
}

//// Reset corner for Optical Flow
int COpticalFlowTracker::ResetCorner()
{
	for(int i=0;i<MAX_CORNERS;i++)
	{
		cornersA1[i].x = 0;
		cornersA1[i].y = 0;
		cornersB1[i].x = 0;
		cornersB1[i].y = 0;
	}
	return 0;
}

//// Construction
COpticalFlowTracker::COpticalFlowTracker()
{
	cornersA1 = new CvPoint2D32f[ MAX_CORNERS ];
	cornersB1 = new CvPoint2D32f[ MAX_CORNERS ];
	ResetCorner();

	m_cornerCount = 0;
	m_objcount = 0;
}

//// Deconstruction
COpticalFlowTracker::~COpticalFlowTracker()
{
	delete []cornersA1;
	delete []cornersB1;
}

//// Return total vehicles
int COpticalFlowTracker::GetObjectCount(int vehicleType)
{
	int numObject = 0;
	numObject = m_objcount;
	return numObject;
}
//// Set bounding line
int COpticalFlowTracker::SetBoundingLine(cv::Point ptStart, cv::Point ptEnd, int iBoundingType)
{
	BoundingLine boundingLine;
	boundingLine.ptStart.x = ptStart.x;
	boundingLine.ptStart.y = ptStart.y;
	boundingLine.ptEnd.x = ptEnd.x;
	boundingLine.ptEnd.y = ptEnd.y;
	boundingLine.iBoundingType = iBoundingType;
	m_lineBounding.push_back(boundingLine);
	return 0;
}

//// Set input image for detect and tracking
int COpticalFlowTracker::SetImage(cv::Mat img)
{
	if(img.empty())
		return -1;
	if(!m_image_cur.empty() )
	{
		if(!m_image_pre.empty())
			m_image_pre.release();
		m_image_pre = m_image_cur.clone();
		m_image_cur.release();
	}
	
	if(1 == img.channels())
		m_image_cur = img.clone();		
	else
	{		
		cv::cvtColor(img,m_image_cur,CV_BGR2GRAY);
	}
	return 0;
}

//// Create a new tracker for a new vehicle
int COpticalFlowTracker::CreateTracker(int cX,int cY,int r, int vehicleType,std::vector<track> &m_trackers)
{
	track curTracker;
	curTracker.centerX = cX;
	curTracker.centerY = cY;
	curTracker.arrCenterX.push_back(cX);
	curTracker.arrCenterY.push_back(cY);
	curTracker.radius  = r;
	curTracker.age = 0;
	
	curTracker.objType = vehicleType;	

	curTracker.vr = 0;
	curTracker.vx = 0;
	curTracker.vy = 0;

	curTracker.length = 1;
	curTracker.isCount= false;

	//curTracker.id = -1;
	curTracker.id = m_objcount++;
	
	m_trackers.push_back(curTracker);
	return 0;
}

//// Init tracker, use when list of tracker is empty
int COpticalFlowTracker::InitTrackersList(std::vector<_t_Moto> motor_list, int vehicleType,std::vector<track> &m_trackers)
{
	if(motor_list.size() <1 )
		return -1;

	if(m_trackers.size() > 1)
	{
		for(int i = m_trackers.size(); i>0; i--)
			m_trackers.erase(m_trackers.begin() + i);
	}
	for(int i=0;i<motor_list.size(); i++)
	{
		CreateTracker(motor_list.at(i).vehicleCenterX,motor_list.at(i).vehicleCenterY,motor_list.at(i).vehicleRadius,vehicleType,m_trackers);
	}
	return 0;
}

//// Run OpticalFlow,
//// Input: only need input image
//// Previous coordinates is global param
int COpticalFlowTracker::RunOpticalFlow(std::vector<track> &m_trackers)
{
	//// If no tracker or no image then return
	if(m_trackers.size() < 1)
	{
		return -1;
	}
	if( m_image_cur.empty() || m_image_pre.empty())
		return -1;
		
	if(m_cornerCount < 1)
	{
		return -1;
	}

	int win_size = 10;
	vector<cv::Point2f> vecCornersA1;
	vector<cv::Point2f> vecCornersB1;
    vector<unsigned char> features_found;    
	vector<float> feature_errors;
		
	cv::Size pyr_sz = cv::Size(m_image_pre.cols + 8, m_image_cur.rows/3);
	cv::Mat image_pre = m_image_pre.clone();
	cv::Mat image_cur = m_image_cur.clone();
	
	cv::Mat image_pyrA = cv::Mat(pyr_sz,cv::DEPTH_MASK_32F);
	cv::Mat image_pyrB = cv::Mat(pyr_sz,cv::DEPTH_MASK_32F);

	for(int i=0;i<m_cornerCount;i++)
	{
		cv::Point2f pt = cv::Point2f(cornersA1[i].x,cornersA1[i].y);
		vecCornersA1.push_back(pt);
	}

	cv::calcOpticalFlowPyrLK(image_pre,image_cur,vecCornersA1,vecCornersB1,features_found,feature_errors);
	m_cornerCount = vecCornersB1.size();
	for(int i=0;i< m_cornerCount;i++)
	{
		cv::Point2f pt = vecCornersB1.at(i);
		cornersB1[i].x = pt.x;
		cornersB1[i].y = pt.y;
	}
	return 0;
}

int COpticalFlowTracker::UpdateUsingDetectedData(track* trackerInput,int xk,int yk,int rk, int vehicleType,std::vector<track> &m_trackers)
{
	if(!(*trackerInput).isCount)
	{
		(*trackerInput).isCount = true;		
		//if(VehicleDirection::DirectionUnknown == iDirection)
		//	(*trackerInput).isCount = false;	//// If not sure, wait to next image
	}

	(*trackerInput).age = 0;
	(*trackerInput).centerX = xk;
	(*trackerInput).centerY = yk;
	(*trackerInput).arrCenterX.push_back(xk);
	(*trackerInput).arrCenterY.push_back(yk);
	(*trackerInput).radius = rk;
	
	return 0;
}

//// Predict then update tracker, use when not detect any car or moto
int COpticalFlowTracker::PredictAndUpdateTracker1(int vehicleType,std::vector<track> &m_trackers)
{
	ResetCorner();
	m_cornerCount = 0;
	for(int i=0;i<m_trackers.size();i++)
	{
		track curTracker = m_trackers.at(i);
		cornersA1[i].x = curTracker.centerX;
		cornersA1[i].y = curTracker.centerY;
		m_cornerCount++;
	}
	int rs = RunOpticalFlow(m_trackers);
	if(0 == rs)
	{	
		for(int i=0;i<m_trackers.size();i++)
		{
			if(!m_trackers.at(i).isCount)
			{
				m_trackers.at(i).isCount = true;
				//if(VehicleDirection::DirectionUnknown == iDirection)
				//	m_trackers.at(i).isCount = false;	//// If not sure, wait to next image
			}
			m_trackers.at(i).centerX = cornersB1[i].x;
			m_trackers.at(i).centerY = cornersB1[i].y;
			m_trackers.at(i).arrCenterX.push_back(cornersB1[i].x);
			m_trackers.at(i).arrCenterY.push_back(cornersB1[i].y);
			m_trackers.at(i).age++;
		}
	}
	return 0;
}

//// Predict then update tracker, use when detect some car or moto but not fit to all tracker
int COpticalFlowTracker::PredictAndUpdateTracker2(int* row_mate, int size2, int vehicleType,std::vector<track> &m_trackers)
{	
	ResetCorner();
	m_cornerCount = 0 ;
	for (int  x=0;x<size2;++x)// tracker can predict
		if (row_mate[x]==0)
		{
			cornersA1[m_cornerCount].x = m_trackers.at(x).centerX;
			cornersA1[m_cornerCount].y = m_trackers.at(x).centerY;
			m_cornerCount++;
		}

	int rs = RunOpticalFlow(m_trackers);
	if(0 == rs)
	{		
		int id = 0;
		for (int  x=0; x<size2; ++x)// tracker can predict
			if (row_mate[x]==0)
			{
				if(!m_trackers.at(x).isCount)
				{
					m_trackers.at(x).isCount = true;
					//if(VehicleDirection::DirectionUnknown == iDirection)
					//	m_trackers.at(x).isCount = false;	//// If not sure, wait to next image
				}
				m_trackers.at(x).centerX = cornersB1[id].x;
				m_trackers.at(x).centerY = cornersB1[id].y;
				m_trackers.at(x).arrCenterX.push_back(cornersB1[id].x);
				m_trackers.at(x).arrCenterY.push_back(cornersB1[id].y);
				m_trackers.at(x).age++;
				id++;
			}
	}
	return 0;
}

int COpticalFlowTracker::UpdateUsingHungarian(std::vector<_t_Moto> motor_list,int vehicleType,std::vector<track> &m_trackers)

{
	int size1 = motor_list.size();
	int size2 = m_trackers.size();
	int size = size1;
	if (size2 > size1) size = size2;

	if(0 == size)
		return 0;

	int** Array = new int*[size];
	int** Result= new int*[size]; 
	int** Atemp = new int*[size];
		
	for(int i=0;i<size;i++)
	{
		Array[i]=new int[size];
		Result[i]=new int[size];
		Atemp[i]=new int[size];
	}
	//// Init conditions
	for (int i=0;i<size;i++)
	{
		for (int k=0;k<size;k++)
		{			
			if (i<size1 && k < size2)
			{			
				Array[i][k] = Ndistance(motor_list.at(i).vehicleCenterX, m_trackers.at(k).centerX, 
										motor_list.at(i).vehicleCenterY, m_trackers.at(k).centerY);
				if (Array[i][k] >= 3*motor_list.at(i).vehicleRadius)
					Array[i][k] = 10000;
			} 
			else
				Array[i][k] = 10000;

			Atemp[i][k] = Array[i][k];
		}			
	}

	//// Solve problem
	int* col_mate=new int[size];//detector
	int* row_mate=new int[size];//tracker
	
	solveAssignmentProblemintRect(Atemp,Result,size,size);
	memset(col_mate,0,size*sizeof(int));
	memset(row_mate,0,size*sizeof(int));
			
	//// Update with detected data
	for (int  y=0;y<size;++y)//detector
		for (int  x=0;x<size;++x)//tracker
			if (Result[y][x]==1 && y<size1 && x<size2)
			{
				if (Array[y][x] < 10000)
				{
					col_mate[y]=1;
					row_mate[x]=1;					
					UpdateUsingDetectedData(&(m_trackers.at(x)), 
											  motor_list.at(y).vehicleCenterX, motor_list.at(y).vehicleCenterY, 
											  motor_list.at(y).vehicleRadius,
											  vehicleType,
											  m_trackers);
				}
			}

	////// Update with predicted data	
	PredictAndUpdateTracker2(row_mate,size2,vehicleType,m_trackers);

	////// Create new tracker
	for (int  y=0;y<size1;++y)//detect moi
		if (col_mate[y] == 0)
			CreateTracker(motor_list.at(y).vehicleCenterX, motor_list.at(y).vehicleCenterY, motor_list.at(y).vehicleRadius, vehicleType,m_trackers);
		
	for(int i=0;i<size;i++)
	{
		delete [] Array[i];	
		delete [] Atemp[i];	
		delete [] Result[i];
	}
				
	delete [] Array;
	delete [] Atemp;
	delete [] Result;
	delete [] col_mate;
	delete [] row_mate;

	return 0;
}

int COpticalFlowTracker::hungarian_assign(										  
										  cv::Mat img_cur, 
										  std::vector<_t_Moto> motor_list,
										  cv::Point pt ,
										  int iVehicleType,
										  std::vector<track> &m_trackers)
{
	int x0 = pt.x;
	int y0 = pt.y;
	int size1 = (int)motor_list.size();		
	int size2 = (int)(m_trackers.size());

	for (int i=0;i<size1;i++)
	{
		motor_list.at(i).vehicleCenterX += x0;
		motor_list.at(i).vehicleCenterY += y0;
	}

	//< 0. Set image, convert image to Gray
	SetImage(img_cur);

	//< 1. Chua co tracker nao
	if (size2 < 1)
	{		
		if(size1 < 1)
			return 0;
   		InitTrackersList(motor_list,iVehicleType,m_trackers);
		return 0;
	}

	//< 2. Khong detect duoc gi
	if (size1 < 1)
	{
		PredictAndUpdateTracker1(iVehicleType,m_trackers);
		return 0;
	}

	//< 3. Su dung thuat toan Hungari
	////http://ranger.uta.edu/~weems/NOTES5311/hungarian.c
	UpdateUsingHungarian(motor_list,iVehicleType,m_trackers );	

	return 0;
}

int COpticalFlowTracker::DrawTracker(cv::Mat img, int x0, int y0, int vehicleType,std::vector<track> &m_trackers) 
{
	char text[50];

	//// Draw rectangle and number of vehicle	
	for (unsigned int i=0;i< m_trackers.size();++i)
	{
		track curTrack = m_trackers.at(i);
		
		int x1 = curTrack.centerX - curTrack.radius + x0;
		int x2 = curTrack.centerX + curTrack.radius + x0;
		int y1 = curTrack.centerY - curTrack.radius + y0;
		int y2 = curTrack.centerY + curTrack.radius + y0;


		if(-1 == m_trackers.at(i).id)
			sprintf(text,"NA");
		else
		{
			if (0 == m_trackers.at(i).age)
				sprintf(text,"%d*",m_trackers.at(i).id); 
			else 
				sprintf(text,"%d",m_trackers.at(i).id);
		}
		
        if (TypeMoto == m_trackers.at(i).objType )
			cv::rectangle(img,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,255,0,0),2,8,0);
        else if (TypeCar == m_trackers.at(i).objType )
			cv::rectangle(img,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,255,255,0),2,8,0);

        if(TypeMoto == m_trackers.at(i).objType)
			cv::putText(img,text,cv::Point(x1,y1),cv::FONT_HERSHEY_COMPLEX,1.0,cv::Scalar(0,255,0,0),1,8);	
        else if (TypeCar == m_trackers.at(i).objType )
			cv::putText(img,text,cv::Point(x1,y1),cv::FONT_HERSHEY_COMPLEX,1.0,cv::Scalar(255,0,255,0),1,8);	
	}
	return 0;
}

#include <time.h>
#include <fstream> 
#include <iostream>

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
static const std::string currentDateTime() {

    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

static const std::string ToString(int objType)
{
    return (objType == TypeMoto)?"moto":"car";
}

extern int g_Count_car;
extern int g_Count_moto;
extern std::string dir_path;
extern std::string cam_ip_adress;

static void WriteStats(track t)
{
    if( t.speed_sign > 0 )
    {
        time_t     now = time(0);
        char type = t.objType;

        struct tm  tstruct;
        char       buf[80];
        tstruct = *localtime(&now);
        strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);

        string filename;
#ifdef linux
        filename = dir_path + "/" + cam_ip_adress + "-" + string(buf) + ".bin";
#endif

#ifdef _WIN32
        filename = dir_path + "\\" + cam_ip_adress + "-" + string(buf) + ".bin";
#endif

        std::ofstream binfile;

        binfile.open (filename.c_str(), ios::app);

        binfile.write( (char*) &now, sizeof(now));
        binfile.write( &type, sizeof(type));

        if( t.objType == TypeCar)
            g_Count_car++;

        if( t.objType == TypeMoto)
            g_Count_moto++;

//        binfile << currentDateTime() << " " << ToString(t.objType) << endl;
        binfile.close();
    }

}

static void ClearTrack(std::vector<track> &m_trackers, int id)
{
	if (m_trackers.at(id).speed_sign > 0)
        WriteStats(m_trackers.at(id));

	m_trackers.at(id).arrCenterX.clear();
	m_trackers.at(id).arrCenterY.clear();
	m_trackers.erase(m_trackers.begin() + id);
}

int COpticalFlowTracker::
DeleteOldTrackers(cv::Mat img_mask,
                  std::vector<track> &m_trackers)
{	
	unsigned int numOfTrack = m_trackers.size();
	unsigned int id = numOfTrack-1;

	for(int i = 0; i < numOfTrack; i++)	
	{		
		track trackTemp = m_trackers.at(id);
		if( (trackTemp.centerX < 0) || (trackTemp.centerX >img_mask.cols - 1 ) ||
			(trackTemp.centerY < 0) || (trackTemp.centerY > img_mask.rows -1) )
		{
            ClearTrack(m_trackers, id);
		}
		else if (trackTemp.age > 10 ||
			img_mask.at<unsigned char>(trackTemp.centerY,trackTemp.centerX) < 255 )
		{	
            ClearTrack(m_trackers, id);
		} 
		id--;
	}
	return 0;
}
