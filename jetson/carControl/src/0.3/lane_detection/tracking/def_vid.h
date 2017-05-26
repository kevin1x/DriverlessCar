//#pragma once

//#ifndef USE_CAMERA_INPUT
////=============================================================================
//#define MAX_MIN_RATE	2

//#define MOTO_ROI_X		120
//#define MOTO_ROI_Y		120
//#define MOTO_ROI_W		500
//#define MOTO_ROI_H		350

//#define MOTO_MIN_SIZE	60
//#define MOTO_MAX_SIZE	MOTO_MIN_SIZE * MAX_MIN_RATE

//#define CAR_ROI_X		120
//#define CAR_ROI_Y		120
//#define CAR_ROI_W		500
//#define CAR_ROI_H		450

//#define CAR_MIN_SIZE	100
//#define CAR_MAX_SIZE	CAR_MIN_SIZE * MAX_MIN_RATE

//#define WINDOW_UPSCALE_RATE		1.1

//#define OUT_VIDEO_FRAME_WIDTH		800
//#define OUT_VIDEO_FRAME_HEIGHT		600

////=============================================================================
//#endif



#pragma once

#ifndef USE_CAMERA_INPUT
//=============================================================================
#define MAX_MIN_RATE	2.0

#define OUT_VIDEO_FRAME_WIDTH		860//1280// 800
#define OUT_VIDEO_FRAME_HEIGHT		480//720 // 450

#define CAR_ROI_X		0.02*OUT_VIDEO_FRAME_WIDTH
#define CAR_ROI_Y		0.30*OUT_VIDEO_FRAME_HEIGHT
#define CAR_ROI_W		0.96*OUT_VIDEO_FRAME_WIDTH
#define CAR_ROI_H		0.69*OUT_VIDEO_FRAME_HEIGHT

#define CAR_MIN_SIZE	90
#define CAR_MAX_SIZE	110

#define MOTO_ROI_X		CAR_ROI_X
#define MOTO_ROI_Y		(CAR_ROI_Y+120)
#define MOTO_ROI_W		CAR_ROI_W
#define MOTO_ROI_H		(CAR_ROI_H-115)

#define MOTO_MIN_SIZE	40
#define MOTO_MAX_SIZE	100


#define WINDOW_UPSCALE_RATE		1.1

//=============================================================================
#endif
