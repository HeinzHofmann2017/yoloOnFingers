/**************************************************
    Projekt:        Collect Fingertip Data
    File:           main.cpp
    Date:           06.04.2017
    Author:         tmendez

    Description:    main-programm for collecting
                    fingertip data

**************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#include "opencv2/opencv.hpp"

// Possible formats (640 x 480 | 1280 x 720 | 1280 x 960 )
#define IM_WIDTH        1280
#define	IM_HEIGHT       960
#define N_CAMS          1

#define NUMB_IMAGES     3

#define PLOT_TIMESLOTS

void delay(void);
void makeDir(const char* dir);

int main(void) {

    std::cout << std::endl << "---------------------- Collect Fingertip Data Application ----------------------" << std::endl << std::endl;

    bool error;
    int id, j;
    unsigned long imCnt;

    unsigned int frameRate = 0;             // frame rate of camera devices
    cv::VideoCapture cameraDevice[N_CAMS];  // camera devices
    cv::Mat camImage[N_CAMS];               // captured images
    cv::Mat convertedImage[N_CAMS];         // converted images
    struct timespec time;                   // time-variables, to measure time between grabbed frames
    double timeLastFrame_ms[N_CAMS], timeCurrentFrame_ms[N_CAMS];
    double deltaTime_ms;
    double timestamps_ms[NUMB_IMAGES+1];

    // initialize the camera devices
    for(id = 0; id<N_CAMS; ++id) {

        // Open the camera device
        cameraDevice[id].open(id);
        if( !cameraDevice[id].isOpened() )
        {
            std::cout << std::endl << "Camera Device " << id << " not initialized successfully." << std::endl << std::endl;
            return 1;
        }

        //Set up the width and height of the camera
        cameraDevice[id].set(cv::CAP_PROP_FRAME_WIDTH,  IM_WIDTH);
        cameraDevice[id].set(cv::CAP_PROP_FRAME_HEIGHT, IM_HEIGHT);

        std::cout << std::endl << "Camera Settings Cam " << id << ": " << std::endl;
        std::cout << "------------------------------" << std::endl;
        std::cout << "Width:  " << cameraDevice[id].get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
        std::cout << "Height: " << cameraDevice[id].get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
        std::cout << "fps:    " << cameraDevice[id].get(cv::CAP_PROP_FPS) << std::endl;
        std::cout << "------------------------------" << std::endl << std::endl;

        if ( frameRate > 0 && frameRate != cameraDevice[id].get(cv::CAP_PROP_FPS) ) {
            std::cout << "Camera Devices have different frame rates." << std::endl << std::endl;
            return 1;
        } else {
            frameRate = cameraDevice[id].get(cv::CAP_PROP_FPS);
        }
    }

    // initialize timestamps
    clock_gettime(CLOCK_REALTIME, &time);
    for(id = 0; id<N_CAMS; ++id) {
        timeLastFrame_ms[id] = time.tv_sec*1.0e3 + time.tv_nsec/1.0e6;
        timeCurrentFrame_ms[id] = time.tv_sec*1.0e3 + time.tv_nsec/1.0e6;
    }
    deltaTime_ms = 0;

    // start cameras
    for(id = 0; id<N_CAMS; ++id) {
        error = !cameraDevice[id].grab();
    }

    // Start main-Loop
   	std::cout << "Collect Images... " << std::endl;

   	clock_gettime(CLOCK_REALTIME, &time);
    timestamps_ms[0] = time.tv_sec*1.0e3 + time.tv_nsec/1.0e6;
    for(imCnt = 0; imCnt<NUMB_IMAGES; ++imCnt) {

        // wait for 1s to make sure the buffer is full
        usleep((imCnt+1)*1.0e6);

        // grab frames
        for(j=0; j<10; ++j){
            for(id = 0; id<N_CAMS; ++id) {
                timeLastFrame_ms[id] = timeCurrentFrame_ms[id];
                error = !cameraDevice[id].grab();
                clock_gettime(CLOCK_REALTIME, &time);
                timeCurrentFrame_ms[id] = time.tv_sec*1.0e3 + time.tv_nsec/1.0e6;
                deltaTime_ms = timeCurrentFrame_ms[id] - timeLastFrame_ms[id];
                std::cout << "deltaTime_ms " << deltaTime_ms << " ms." << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "Done. " << std::endl << std::endl;


        // get time-stamp
        clock_gettime(CLOCK_REALTIME, &time);
        timestamps_ms[imCnt+1] = time.tv_sec*1.0e3 + time.tv_nsec/1.0e6;
	}

	cameraDevice[0].retrieve(camImage[0]);
	cv::convertScaleAbs(camImage[0], convertedImage[0], 0.06226);
	std::stringstream picName;
    picName << "./test.png";
    cv::imwrite(picName.str(), convertedImage[0]);

	// calculate achieved frame-rate
	std::cout << "Achieved framerate: " << imCnt*1000/(timestamps_ms[imCnt]-timestamps_ms[0]) << " fps" << std::endl << std::endl;

	// release camera devices
	for(id = 0; id<N_CAMS; ++id) {
        cameraDevice[id].release();
	}

    return 0;
}



void delay(void)
{
    volatile unsigned int i,j;

    for (i = 0; i < 50000; ++i) {
        for (j = 0; j < 10000; ++j) {}
    }
}

