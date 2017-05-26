#include "def.h"
#include "stdio.h"
#include <vector>

#pragma once

	enum VehicleType
	{
		TypeMoto= 0,
		TypeCar = 1
	};

	enum VehicleSide
	{
		SideTop,
		SideBottom,
		SideLeft,
		SideRight
	};

	enum BoundingType
	{
		TypeTop = 0,
		TypeBottom = 1,
		TypeLeft = 2,
		TypeRight
	};

	typedef struct _t_Moto_
	{
		int vehicleCenterX;
		int vehicleCenterY;
		int vehicleSizeW; // vehicle's width
		int vehicleSizeH; // vehicle's height
		int vehicleRadius;
		float vehicleAngle;
	} _t_Moto;

	typedef struct _track
	{
		int id;
		int age;
		
		int objType;
		int objDirection;

		int vx,vy,vr;
		int length;
		
		int centerX;
		int centerY;
		int radius;
		
		bool isCount;

		std::vector<int> arrCenterX;
		std::vector<int> arrCenterY;

		int firstPosX;
		int firstPosY;
		int oldCenterX;
		int oldCenterY;
		float dy;
		float dx;
		int speed_sign;

		_track()
		{
			dx = -1;
			dy = -1;
			oldCenterX = 9999;
			oldCenterY = 9999;
			firstPosX = -1;
			firstPosY = -1;
			speed_sign = -1;
		}

	}track ;


