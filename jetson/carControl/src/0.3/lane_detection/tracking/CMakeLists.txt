
cmake_minimum_required(VERSION 2.8)

project( VehicleTrackingLib )

set( lib1 vehicleTracking)

set( source_files
	COpticalFlowTracker.cpp
	CVehicleCounter.cpp
	CVehicleDetector.cpp
    hungarian.cpp

	COpticalFlowTracker.h
	CVehicleCounter.h
	CVehicleDetector.h
	hungarian.h
	Param.h
    resource.h
    def.h
    def_cam.h

)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library( ${lib1} STATIC ${source_files} )

target_link_libraries( ${lib1} ${OpenCV_LIBS} )
