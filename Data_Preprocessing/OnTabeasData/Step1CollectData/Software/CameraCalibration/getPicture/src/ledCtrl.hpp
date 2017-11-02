/****************************************************
    Projekt:        Collect Fingertip Data
    File:           ledCtrl.hpp
    Date:           06.04.2017
    Author:         tmendez

    Description:    UART Interface for controlling
                    the led of the Led Illumination

****************************************************/

#ifndef LEDCRTL_H_
#define LEDCRTL_H_


class LedCtrl
{
    public:

        enum ledType {UV, IR, WHITE};

        // Initializes the serialport
        LedCtrl(const char* desiredPort);

        // Closes the serialport
        virtual ~LedCtrl();

        // get if serialport is ready
        bool inline getSerPortReady(){
            return serPortReady;
        }

        // Sets the Duty-Cycle of the leds
        // return:  true if not successful
        bool ledSet(ledType type, unsigned char dutyCycle);

    private:
        struct sp_port *serPort;                    // pointer of the serial port-instance
        bool serPortReady;                          // is set to true, when the initialization was successful
        static const char inputBuffSize = 2;        // size of the input buffer
        static const char outputBuffSize = 2;       // size of the output buffer
        unsigned char inputBuff[inputBuffSize];     // input buffer
        unsigned char outputBuff[outputBuffSize];   // output buffer
        enum commands{ UV_ON=0x11,                  // commands to send to the serial port
                       IR_ON=0x22,
                       WHITE_ON=0x44,
                       OFF = 0x88} ;

};



#endif /* LEDCRTL_H_ */


