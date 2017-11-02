/**************************************************
    Projekt:        Led Illumination
    File:           pwm.h
    Date:           30.03.2017
    Author:         tmendez

    Description:    PWM Driver for MKL26Z4

**************************************************/

#ifndef PWM_H_
#define PWM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {UV, IR, WHITE} ledType_t;

/* Initialises the PWM-Module
*/
void pwm_init(void);

/* Sets the Duty-Cycle of the PWM-Signal
*  param:   ledType:     led type of which the PWM is changed
*           dutyCycle:   duty-cycle to which the PWM is set (0...127)
*/
void pwm_set(ledType_t ledType, uint8_t dutyCycle);


#ifdef __cplusplus
}
#endif
#endif /* PWM_H_ */
