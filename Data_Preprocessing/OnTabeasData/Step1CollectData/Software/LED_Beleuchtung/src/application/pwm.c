/**************************************************
    Projekt:        Led Illumination
    File:           pwm.c
    Date:           30.03.2017
    Author:         tmendez

    Description:    PWM Driver for MKL26Z4

**************************************************/

#include <stdint.h>

#include "MKL26Z4.h"
#include "pwm.h"


#define PWM_CLOCK   (48000000/16)
#define PWM_FREQ    5000
#define CNT_LIMIT   ((PWM_CLOCK)/PWM_FREQ)      // Max: 2^16-1

void pwm_init(void) {

    // Enable clock to port pins used by PWM
    SIM->SCGC5 =    SIM_SCGC5_PORTA_MASK |
                    SIM_SCGC5_PORTB_MASK |
                    SIM_SCGC5_PORTC_MASK |
                    SIM_SCGC5_PORTD_MASK |
                    SIM_SCGC5_PORTE_MASK;

    // Enable clock to PWM
    SIM_SCGC6 = ( SIM_SCGC6_TPM1_MASK | SIM_SCGC6_TPM2_MASK );
    SIM->SOPT2 |= SIM_SOPT2_TPMSRC(1);

    // Set Pin function to PWM
    PORTE->PCR[20] = PORT_PCR_MUX(3); //UV-led
    PORTE->PCR[21] = PORT_PCR_MUX(3); //White-led
    PORTE->PCR[22] = PORT_PCR_MUX(3); //IR-led

    // Setup PWM-Module in Edge-Aligned PWM Mode (Reference Manual 31.4.6)
    TPM1->SC |= TPM_SC_CMOD(0);         // disable counter
    TPM1->SC |= TPM_SC_PS(4);           // clock divide by 2^4 = 16 (max 2^7)
    TPM1->SC &= ~TPM_SC_CPWMS_MASK;     // up counting mode
    TPM1->MOD = CNT_LIMIT-1;            // set period

    TPM2->SC |= TPM_SC_CMOD(0);         // disable counter
    TPM2->SC |= TPM_SC_PS(4);           // clock divide by 2^4 = 16 (max 2^7)
    TPM2->SC &= ~TPM_SC_CPWMS_MASK;     // up counting mode
    TPM2->MOD = CNT_LIMIT-1;            // set period

    // setup Edge-Aligned PWM Mode with logic 1 = high for UV-led
    TPM1->CONTROLS[0].CnSC = (TPM_CnSC_ELSB_MASK | TPM_CnSC_MSB_MASK);
    TPM1->CONTROLS[0].CnV = 0;          // set duty-cycle to 0%

    // setup Edge-Aligned PWM Mode with logic 1 = high for White-led
    TPM1->CONTROLS[1].CnSC = (TPM_CnSC_ELSB_MASK | TPM_CnSC_MSB_MASK);
    TPM1->CONTROLS[1].CnV = 0;          // set duty-cycle to 0%

    // setup Edge-Aligned PWM Mode with logic 1 = high for IR-led
    TPM2->CONTROLS[0].CnSC = (TPM_CnSC_ELSB_MASK | TPM_CnSC_MSB_MASK);
    TPM2->CONTROLS[0].CnV = 0;          // set duty-cycle to 0%

    // start counters
    TPM1->SC |= TPM_SC_CMOD(1);
    TPM2->SC |= TPM_SC_CMOD(1);
}


void pwm_set(ledType_t ledType, uint8_t dutyCycle){

    if (ledType == UV) {
        TPM1->CONTROLS[0].CnV = (CNT_LIMIT*dutyCycle/127);
    } else if (ledType == IR) {
        TPM2->CONTROLS[0].CnV = (CNT_LIMIT*dutyCycle/127);
    } else if (ledType == WHITE) {
        TPM1->CONTROLS[1].CnV = (CNT_LIMIT*dutyCycle/127);
    }

}
