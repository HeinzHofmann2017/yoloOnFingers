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

#include "ledCtrl.hpp"
#include "opencv2/opencv.hpp"

// Possible formats (640 x 480 | 1280 x 720 | 1280 x 960 )
#define IM_WIDTH        1280
#define	IM_HEIGHT       960
#define N_CAMS          4

#define PWM_UV          150
#define PWM_IR          0
#define PWM_WHITE       80

#define MAX_PATH_LEN    320

#define N_IMAGES        300


void delay(void);
void makeDir(const char* dir);

int main(void) {

    std::cout << std::endl << "---------------------- Get Picture of Cameras ----------------------" << std::endl << std::endl;

    bool error;
    int id, j, k, trials;
    char keyPressed;

    // type of pictures to get
    char calType;
    int camNr = 0;

    // position parameters
    char side;
    char orientation;
    int position;
    int angle;
    int direction;
    char light[2] = {'w', 'u'};

    const char* desiredPort = "/dev/ttyUSB0";           // name of the serial port to use
    unsigned int frameRate = 0;                         // frame rate of camera devices
    cv::VideoCapture cameraDevice[N_CAMS];              // camera devices
    cv::Mat camImage[N_CAMS][2];                        // captured images
    cv::Mat convertedImage[N_CAMS];                     // converted images
    LedCtrl::ledType illumination = LedCtrl::WHITE;     // type of illumination to turn on


    // get type of calibration pictures to collect
    std::cout << std::endl << "Which type of calibration pictures are collected?" << std::endl;
    std::cout << "Calibration pictures (i = intrinsic | e = extrinsic):  ";
    std::cin >> calType;
    if (calType != 'i' && calType != 'e') {
        std::cout  << std::endl << "Type of calibration pictures to collect must be 'i' or 'e' !" << std::endl << std::endl ;
        return 1;
    }

    // make sure the path to save the pictures exists
    const char rootDir[] = "/media/tabea/DATA/00_Masterarbeit_FingerTracking/MA_fingertipTracking/Software/CameraCalibration/calibrationPics/";
    if ( 20 + sizeof(rootDir)/sizeof(char) > MAX_PATH_LEN ) {
        std::cout  << std::endl << "Root directory is to long. Maximal number of characters is " << MAX_PATH_LEN - 20 << "." << std::endl << std::endl ;
        return 1;
    }
    char savePath[N_CAMS][MAX_PATH_LEN];
    for(id = 0; id < N_CAMS; ++id){
        for(j = 0; j < MAX_PATH_LEN; ++j){
            // copy root directory
            savePath[id][j] = rootDir[j];
            if ( savePath[id][j] == '\0' ){
                break;
            }
        }
        if (calType == 'i') {
            char temp[] = "/Camera_?/intrinsic/";
            temp[8] = (char)(id + '0');
            strcat(savePath[id], temp);
        } else {
            char temp[] = "/Camera_?/extrinsic/";
            temp[8] = (char)(id + '0');
            strcat(savePath[id], temp);
        }
        makeDir(savePath[id]);
    }

    // initialize the serial port
    LedCtrl myLedCtrl(desiredPort);
    error = !myLedCtrl.getSerPortReady();
    if (error) {
        std::cout << std::endl << "Serial port not initialized successfully." << std::endl << std::endl;
        return 1;
    }

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

    // collect pictures for intrinsic parameters
    if(calType == 'i'){
        std::cout << "Camera Number (0...3):  ";
        std::cin >> camNr;
        if (camNr <  0 || camNr > 3){
            std::cout << std::endl << "No valid camera number." << std::endl << std::endl;
            return 1;
        }

        // clear buffers of the camera devices
        error = !cameraDevice[camNr].grab();
        usleep(5.0e6);      // wait for 1s to make sure the buffer is full
        for(j=0; j<4; ++j){
            error = !cameraDevice[camNr].grab();
        }

        // Turn on led illumination
        trials = 0;
        do {
            error = myLedCtrl.ledSet(LedCtrl::UV, PWM_UV);
            ++trials;
        } while ( error && trials < 5 );
        if ( error ){
            std::cout << "Change of led illumination was not successful." << std::endl;
        }

        // get the images
        for(j = 0; j<N_IMAGES; ++j) {
            for (k=0; k<4; k++) {
                error = !cameraDevice[camNr].grab();
                if (error){
                    std::cout << "No frame grabbed from cam " << camNr << ". Check whether the camera is free." << std::endl;
                }
            }
            error = !cameraDevice[camNr].retrieve(camImage[camNr][0]);
            if ( error || camImage[camNr][0].empty() ){
                std::cout << "No frame retrieved from cam " << camNr << ". Check whether the camera is free." << std::endl;
            }

            // Convert to 8 Bit: Scale the 12 Bit (4096) Pixels into 8 Bit(255) (255/4096)= 0.06226
            cv::convertScaleAbs(camImage[camNr][0], convertedImage[camNr], 0.06226);

            // save captured image
            std::stringstream picName;
            picName << savePath[camNr] << "calBoard_" << j <<".png";
            error = !cv::imwrite(picName.str(), convertedImage[camNr]);
            if (error) {
                std::cout << "Could not save image " << picName.str() << "." << std::endl;
            }

            // show captured image
            cv::namedWindow("Captured Image", cv::WINDOW_AUTOSIZE);
            cv::imshow("Captured Image",convertedImage[camNr]);

            keyPressed = cv::waitKey(1); //Waits for a user input to quit the application

            if(keyPressed == 'q'  || keyPressed == 'Q' ) {
                cv::destroyAllWindows();
                break;
            }
            usleep(500.0e3);
        }

        // turn of the illumination
        myLedCtrl.ledSet(LedCtrl::WHITE, 0);

    } else {
        // collect pictures for extrinsic parameters
        while(1){

            // read in position parameters
            std::cout << std::endl << "Please type in the position parameters of the calibration board." << std::endl;
            std::cout << "To quit type 'q'" << std::endl;
            std::cout << "Side (l = left | r = right):  ";
            std::cin >> side;
            if (side == 'q') {
                std::cout << std::endl;
                break;
            }
            std::cout << "Orientation (h = horizontal | v = vertical):  ";
            std::cin >> orientation;
            std::cout << "Position (0...8):  ";
            std::cin >> position;
            std::cout << "Angle (-45 | 0 | 45):  ";
            std::cin >> angle;
            std::cout << "Direction (-1 | 1):  ";
            std::cin >> direction;

            // clear buffers of the camera devices
            for(id = 0; id<N_CAMS; ++id) {
                error = !cameraDevice[id].grab();
            }
            usleep(1.0e6);      // wait for 1s to make sure the buffer is full
            for(j=0; j<4; ++j){
                for(id = 0; id<N_CAMS; ++id) {
                    error = !cameraDevice[id].grab();
                }
            }

            // Turn on led illumination WHITE
            trials = 0;
            do {
                error = myLedCtrl.ledSet(LedCtrl::WHITE, PWM_WHITE);
                ++trials;
            } while ( error && trials < 5 );
            if ( error ){
                std::cout << "Change of led illumination was not successful." << std::endl;
            }

            // grab next frame of every camera device
            for(j=0; j<4; ++j){
                for(id = 0; id<N_CAMS; ++id) {
                    error = !cameraDevice[id].grab();
                    if (error){
                        std::cout << "No frame grabbed from cam " << id << ". Check whether the camera is free." << std::endl;
                    }
                }
            }

            // Turn on led illumination UV
            trials = 0;
            do {
                error = myLedCtrl.ledSet(LedCtrl::UV, PWM_UV);
                ++trials;
            } while ( error && trials < 5 );
            if ( error ){
                std::cout << "Change of led illumination was not successful." << std::endl;
            }

            // get the "WHITE" frames
            for(id = 0; id<N_CAMS; ++id) {
                error = !cameraDevice[id].retrieve(camImage[id][0]);
                if ( error || camImage[id][0].empty() ){
                    std::cout << "No frame retrieved from cam " << id << ". Check whether the camera is free." << std::endl;
                }
            }

            // grab next frame of every camera device
            for(j=0; j<4; ++j){
                for(id = 0; id<N_CAMS; ++id) {
                    error = !cameraDevice[id].grab();
                    if (error){
                        std::cout << "No frame grabbed from cam " << id << ". Check whether the camera is free." << std::endl;
                    }
                }
            }

            // turn off the illumination
            myLedCtrl.ledSet(LedCtrl::UV, 0);


            // get the "UV" frames
            for(id = 0; id<N_CAMS; ++id) {
                error = !cameraDevice[id].retrieve(camImage[id][1]);
                if ( error || camImage[id][1].empty() ){
                    std::cout << "No frame retrieved from cam " << id << ". Check whether the camera is free." << std::endl;
                }
            }

            for(k=0; k<2; k++){

                //Convert to 8 Bit: Scale the 12 Bit (4096) Pixels into 8 Bit(255) (255/4096)= 0.06226
                for(id = 0; id<N_CAMS; ++id) {
                    cv::convertScaleAbs(camImage[id][k], convertedImage[id], 0.06226);
                }

                // save captured images
                for(id = 0; id<N_CAMS; ++id) {

                    std::stringstream picName;
                    picName << savePath[id] << "calBoard_" << side << "_"
                                                           << orientation << "_"
                                                           << position << "_"
                                                           << angle << "_"
                                                           << direction<< "_"
                                                           << light[k] <<".png";
                    error = !cv::imwrite(picName.str(), convertedImage[id]);
                    if (error) {
                        std::cout << "Could not save image " << picName.str() << "." << std::endl;
                    }
                }
            }
        }
    }


	// release camera devices
	for(id = 0; id<N_CAMS; ++id) {
        cameraDevice[id].release();
	}

    return 0;
}



void makeDir(const char* dir){

    bool madeDir = false;
    char temp[MAX_PATH_LEN];

    // build up directory path step by step
    for(int i = 0; i < MAX_PATH_LEN; ++i){

        temp[i] = dir[i];

        if( temp[i] == '/' ){
            temp[i+1] = '\0';
            struct stat st = {0};
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

