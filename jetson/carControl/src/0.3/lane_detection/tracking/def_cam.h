#pragma once

#ifdef USE_CAMERA_INPUT
//=============================================================================
#define MAX_MIN_RATE	2

#define MOTO_ROI_X		120
#define MOTO_ROI_Y		120
#define MOTO_ROI_W		500
#define MOTO_ROI_H		350

#define MOTO_MIN_SIZE	60
#define MOTO_MAX_SIZE	MOTO_MIN_SIZE * MAX_MIN_RATE

#define CAR_ROI_X		120
#define CAR_ROI_Y		120
#define CAR_ROI_W		500
#define CAR_ROI_H		450

#define CAR_MIN_SIZE	100
#define CAR_MAX_SIZE	CAR_MIN_SIZE * MAX_MIN_RATE

#define WINDOW_UPSCALE_RATE		1.1

#define OUT_VIDEO_FRAME_WIDTH		800
#define OUT_VIDEO_FRAME_HEIGHT		600
//=============================================================================
#endif
