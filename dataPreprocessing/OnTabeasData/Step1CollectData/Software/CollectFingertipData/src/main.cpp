/**************************************************
    Projekt:        Collect Fingertip Data
    File:           main.cpp
    Date:           06.04.2017
    Author:         tmendez

    Description:    main-programm for collecting
                    fingertip data

**************************************************/

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Barrier.hpp"
#include "ledCtrl.hpp"
#include "opencv2/opencv.hpp"

#include "stacktrace.h"
#include <signal.h>

// Possible formats (640 x 480 | 1280 x 720 | 1280 x 960 )
#define IM_WIDTH        1280
#define	IM_HEIGHT       960
#define N_CAMS          4

#define PWM_UV          100     // 0 ... 127
#define PWM_IR          0       // 0 ... 127
#define PWM_WHITE       40      // 0 ... 127

#define MAX_TRIALS      5

#define MAX_PATH_LEN    320

#define NUMB_IMAGES     6000

//#define DEBUG

void delay(void);
void makeDir(std::string& dir);
void camThreadFunc(int camId, cv::VideoCapture cameraDevice, std::string& rootDir, Barrier& preIllumChangeBarrier, Barrier& preGetFrameBarrier);

// main-function
 int main(int /*argc*/, char** /*argv*/) {
    signal(SIGSEGV, stacktrace);

    std::cout << std::endl << "---------------------- Collect Fingertip Data Application ----------------------" << std::endl << std::endl;

    bool error;
    int id, trials;
    unsigned long imCnt;

    const char* desiredPort = "/dev/ttyUSB0";           // name of the serial port to use
    struct timespec time;                               // time-variables, to measure time between grabbed frames
    double timestamps_ms[NUMB_IMAGES+1];

    cv::VideoCapture cameraDevice[N_CAMS];              // camera devices
    std::thread camThread[N_CAMS];                      // threads to handle cameras
    Barrier preIllumChangeBarrier(N_CAMS+1);            // barrier for thread synchronization
    Barrier preGetFrameBarrier(N_CAMS+1);               // barrier for thread synchronization

    std::string rootDir = "/media/heinz/Elements/Daten/indexfinger_right";

    // initialize the serial port
    LedCtrl myLedCtrl(desiredPort);
    error = !myLedCtrl.getSerPortReady();
    if (error) {
        std::cout << std::endl << "Serial port not initialized successfully." << std::endl << std::endl;
        return 1;
    }

    // initialize timestamps
    clock_gettime(CLOCK_REALTIME, &time);


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
    }

    // start the camera-threads
    std::cout << "Start the camera devices... " << std::endl;
    for(id = 0; id<N_CAMS; ++id) {
        camThread[id] = std::thread(camThreadFunc, id, cameraDevice[id], std::ref(rootDir), std::ref(preIllumChangeBarrier), std::ref(preGetFrameBarrier));
    }

    // wait for cameras to be ready
    preIllumChangeBarrier.await();

    // Start main-Loop
   	std::cout << "Collect Images... " << std::endl;

   	clock_gettime(CLOCK_REALTIME, &time);
    timestamps_ms[0] = time.tv_sec*1.0e3 + time.tv_nsec/1.0e6;
    for(imCnt = 0; imCnt<NUMB_IMAGES; ++imCnt) {

	    // change led illumination to WHITE
	    trials = 0;
	    do {
            error = myLedCtrl.ledSet(LedCtrl::WHITE, PWM_WHITE);
            ++trials;
	    } while ( error && trials < MAX_TRIALS );
#ifdef DEBUG
        if ( error ){
            std::cout << "Change of led illumination to WHITE was not successful." << std::endl;
        }
        std::cout << "Changed to WHITE" << std::endl;
#endif // DEBUG

        // wait for cameras to be ready
        preGetFrameBarrier.await();
        preIllumChangeBarrier.await();

	    // change led illumination to UV
	    trials = 0;
	    do {
            error = myLedCtrl.ledSet(LedCtrl::UV, PWM_UV);
            ++trials;
	    } while ( error && trials < MAX_TRIALS );
#ifdef DEBUG
        if ( error ){
            std::cout << "Change of led illumination to UV was not successful." << std::endl;
        }
        std::cout << "Changed to UV" << std::endl;
#endif // DEBUG

        // wait for cameras to be ready
        preGetFrameBarrier.await();
        preIllumChangeBarrier.await();

        // get time-stamp
        clock_gettime(CLOCK_REALTIME, &time);
        timestamps_ms[imCnt+1] = time.tv_sec*1.0e3 + time.tv_nsec/1.0e6;
	}
    std::cout << "Done. " << std::endl << std::endl;

	// calculate achieved frame-rate
	std::cout << "Achieved framerate: " << imCnt*1000/(timestamps_ms[imCnt]-timestamps_ms[0]) << " fps" << std::endl << std::endl;

    // turn of the illumination
    myLedCtrl.ledSet(LedCtrl::UV, 0);
    myLedCtrl.ledSet(LedCtrl::IR, 0);
    myLedCtrl.ledSet(LedCtrl::WHITE, 0);

    // save time-stamps to file
    std::ofstream myfile;
    std::string filePath = rootDir + "/timestamps.txt";

    myfile.open(filePath.c_str());
    if (myfile.is_open()) {
        myfile << "Timestamps of collected images in milliseconds: " << std::endl << std::endl;
        for(imCnt = 0; imCnt<NUMB_IMAGES; ++imCnt)
        {
            myfile << std::setw(3) << imCnt << ": "
                   << std::setw( 12 ) << std::right << std::fixed << std::setprecision(4)
                   << timestamps_ms[imCnt+1]- timestamps_ms[0] << " ms" << std::endl;
        }
        myfile.close();
    } else {
        std::cout << "Unable to open file: " << filePath << std::endl;
    }

    // Save captures images
    std::cout << "Save images... " << std::endl;
	for(id = 0; id<N_CAMS; ++id) {
        camThread[id].join();   // wait for threads to be done
	}
    std::cout << "Done. " << std::endl << std::endl;

    return 0;
}



// Thread-Function to handle cameras
void camThreadFunc(int camId, cv::VideoCapture cameraDevice, std::string& rootDir, Barrier& preIllumChangeBarrier, Barrier& preGetFrameBarrier){

    bool error;
    unsigned long camImCnt;

    // thread variables
    cv::Mat camImageWhite[NUMB_IMAGES];     // captured WHITE images
    cv::Mat camImageUV[NUMB_IMAGES];        // captured UV images
    cv::Mat convertedImageWhite;            // converted WHITE images
    cv::Mat convertedImageUV;               // converted UV images

    // make sure the path to save the pictures exists
    std::string saveDirWhite = rootDir + "/Camera_" + std::to_string(camId) + "/WHITE/";
    std::string saveDirUV = rootDir + "/Camera_" + std::to_string(camId) + "/UV/";
    makeDir(saveDirWhite);
    makeDir(saveDirUV);

    preIllumChangeBarrier.await();

    // Start main-Loop
    for(camImCnt = 0; camImCnt<NUMB_IMAGES; ++camImCnt) {

        // wait for go-signal
        preGetFrameBarrier.await();

        // grab the next "WHITE" frame
        for(int j=0; j<2; ++j){     // one frame is skipped, since the illumination needs to have changed
            error = !cameraDevice.grab();
#ifdef DEBUG
            if (error){
                std::cout << "No frame grabbed from cam " << camId << ". Check whether the camera is free." << std::endl;
            }
#endif // DEBUG
        }
#ifdef DEBUG
        std::cout << "    cam " << camId << ": grabbed white." << std::endl;
#endif // DEBUG

        preIllumChangeBarrier.await();

        // get the grabbed "WHITE" frame
        error = !cameraDevice.retrieve(camImageWhite[camImCnt]);
#ifdef DEBUG
        if ( error || camImageWhite[camImCnt].empty() ){
            std::cout << "No frame retrieved from cam " << camId << ". Check whether the camera is free." << std::endl;
        }
#endif // DEBUG

        // wait for go-signal
        preGetFrameBarrier.await();

        // grab the next "UV" frame
        for(int j=0; j<2; ++j){     // one frame is skipped, since the illumination needs to have changed
            error = !cameraDevice.grab();
 #ifdef DEBUG
           if (error){
                std::cout << "No frame grabbed from cam " << camId << ". Check whether the camera is free." << std::endl;
            }
#endif // DEBUG
        }
#ifdef DEBUG
        std::cout << "    cam " << camId << ": grabbed uv." << std::endl;
#endif // DEBUG

        preIllumChangeBarrier.await();

        // get the grabbed "UV" frame
        error = !cameraDevice.retrieve(camImageUV[camImCnt]);
#ifdef DEBUG
        if ( error || camImageUV[camImCnt].empty() ){
            std::cout << "No frame retrieved from cam " << camId << ". Check whether the camera is free." << std::endl;
        }
#endif // DEBUG

    }

	// release camera devices
    cameraDevice.release();

    // Save captures images
    for(camImCnt = 0; camImCnt<NUMB_IMAGES; ++camImCnt)
	{
        //Convert to 8 Bit: Scale the 12 Bit (4096) Pixels into 8 Bit(255) (255/4096)= 0.06226
        cv::convertScaleAbs(camImageWhite[camImCnt], convertedImageWhite, 0.06226);
        cv::convertScaleAbs(camImageUV[camImCnt], convertedImageUV, 0.06226);

        // save captured images
        std::string picNameWhite = saveDirWhite + "pic" + std::to_string(camImCnt) + ".png";
        error = !cv::imwrite(picNameWhite, convertedImageWhite);
        if (error) {
            std::cout << "Could not save image " << picNameWhite << "." << std::endl;
        }

        std::string picNameUV = saveDirUV + "pic" + std::to_string(camImCnt) + ".png";
        error = !cv::imwrite(picNameUV, convertedImageUV);
        if (error) {
            std::cout << "Could not save image " << picNameUV << "." << std::endl;
        }
	}

}



void makeDir(std::string& dir){

    bool madeDir = false;
    char temp[dir.length()+1];
    struct stat st;
    memset(&st, 0, sizeof(st));

    // build up directory path step by step
    for(unsigned int i = 0; i < dir.length(); ++i){

        temp[i] = dir[i];

        if( temp[i] == '/' ){
            temp[i+1] = '\0';
            memset(&st, 0, sizeof(st));
            if (stat(temp, &st) == -1) {
                mkdir(temp, 0700);
                madeDir = true;
            }
        }
    }

    if (madeDir) {
        std::cout << "Created directory " << dir << std::endl;
    }

}



void delay(void)
{
    volatile unsigned int i,j;

    for (i = 0; i < 50000; ++i) {
        for (j = 0; j < 10000; ++j) {}
    }
}

