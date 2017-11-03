/****************************************************
    Projekt:        Collect Fingertip Data
    File:           ledCtrl.cpp
    Date:           06.04.2017
    Author:         tmendez

    Description:    UART Interface for controlling
                    the led of the Led Illumination

****************************************************/

#include <iostream>
#include <libserialport.h>

#include "ledCtrl.hpp"

#define TIMEOUT_VALUE   5000000

LedCtrl::LedCtrl(const char* desiredPort){

    struct sp_port **ports;
    sp_return error;
    serPortReady = false;

    // find all available serial ports
    error = sp_list_ports(&ports);
    if (error == SP_OK) {
            std::cout << std::endl << "Serial Ports: " << std::endl;
            std::cout << "------------------------------" << std::endl;
            for (int i = 0; ports[i]; i++) {
                std::cout << "Found port: '" << sp_get_port_name(ports[i]) << "'" << std::endl;
            }
            std::cout << "------------------------------" << std::endl << std::endl;
            sp_free_port_list(ports);
    } else {
        std::cout << "No serial devices detected" << std::endl << std::endl;
        return;
    }

    // initialize desired port
    error = sp_get_port_by_name(desiredPort,&serPort);
    if (error == SP_OK) {
        std::cout << "Opening serial port " << sp_get_port_name(serPort) << std::endl;
        error = sp_open(serPort,SP_MODE_READ_WRITE );
        if (error == SP_OK) {
            sp_set_baudrate(serPort,38400);
            sp_set_bits(serPort,8);
            sp_set_parity(serPort,SP_PARITY_ODD);
            sp_set_stopbits (serPort, 1);
            sp_set_flowcontrol (serPort,SP_FLOWCONTROL_NONE);
            sp_flush(serPort, SP_BUF_BOTH);

        } else {
            std::cout << "Error opening serial device" << std::endl;
            return;
        }
    } else {
        std::cout << "Error finding serial device" << std::endl;
        return;
    }

    serPortReady = true;    // Initialization was successful
}



LedCtrl::~LedCtrl(){
    sp_close(serPort);
    std::cout << "Closing serial port " << std::endl;
}



bool LedCtrl::ledSet(ledType type, unsigned char dutyCycle){

    unsigned long timeout = 0;
    int numByte = 0;

    // flush buffers
    sp_flush(serPort, SP_BUF_BOTH);

    // create command
    outputBuff[1] = dutyCycle;
    if (dutyCycle > 0) {
        switch (type){
            case UV:
                outputBuff[0] = UV_ON;
                break;
            case IR:
                outputBuff[0] = IR_ON;
                break;
            case WHITE:
                outputBuff[0] = WHITE_ON;
                break;
            default:
                outputBuff[0] = OFF;
                break;
        }
    } else {
        outputBuff[0] = OFF;
    }

    // send command
    numByte = sp_nonblocking_write(serPort, outputBuff, 2);
    if (numByte < 0){
        return true;   // error occurred
    }
    timeout = 0;
    while(sp_output_waiting(serPort) > 0 && timeout<TIMEOUT_VALUE){
        timeout++;
    }
    if (timeout >= TIMEOUT_VALUE){
        return true;   // timeout occurred
    }

    // receive answer
    timeout = 0;
    while(sp_input_waiting(serPort) < 2 && timeout<TIMEOUT_VALUE){
        timeout++;
    }
    if (timeout >= TIMEOUT_VALUE){
        return true;   // timeout occurred
    }
    numByte = sp_nonblocking_read(serPort,inputBuff,2);
    if (numByte < 2){
        return true;   // error occurred
    }

    // evaluate answer
    if ( inputBuff[0] == 'e' ) {
        return true;   // error occurred
    } else if ( outputBuff[0] != inputBuff[0] || outputBuff[1] != inputBuff[1] ) {
        return true;   // error occurred
    }

    return false; // set led successful
}

