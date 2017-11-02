/**************************************************
    Projekt:        Led Illumination
    File:           main.c
    Date:           30.03.2017
    Author:         tmendez

    Description:    main-programm for turning
                    leds on an off via UART

**************************************************/

#include <MKL26Z4.h>
#include <MKL_EXT.h>

#include "uart.h"
#include "pwm.h"

#define LED_YELLOW  PIN_INV(PB,17)
#define LED_RED     PIN_INV(PB,18)
#define LED_GREEN   PIN_INV(PB,19)

#define CODE_0      PIN(PC,8)
#define CODE_1      PIN(PC,9)
#define CODE_2      PIN(PC,10)
#define CODE_3      PIN(PC,11)

void gpio_init(void);
void delay(void);

enum commands {UV_ON=0x81, IR_ON=0x82, WHITE_ON=0x84, OFF = 0x88};

int main(void)
{
    uint8_t error = 0;
    uint8_t timeout = 0;
    uint8_t led = 0;
    uint8_t pwmValue = 0;
    uint8_t valid = 0;
    uint8_t master = 0;

    gpio_init();
    pwm_init();
    uart_init(1,38400);
    uart_init(2,115200);
    master = ~INPUT_GET(CODE_0);

    // indicate master or slave
    if ( master == 1 ){
        OUTPUT_SET(LED_RED,POSITION_ON);
    } else {
        OUTPUT_SET(LED_YELLOW,POSITION_ON);
    }


    while(1) {

        if ( master == 1 ) {
            // Master listens at UART1 and forwards the command at UART2
            do {
                do {
                    // wait for data to receive at UART1
                    error = uart_rxChar(1, &led);
                }while(error);
                OUTPUT_SET(LED_GREEN,POSITION_ON);
                uart_rxChar(1, &pwmValue);
                pwmValue &= 0x7F;
                OUTPUT_SET(LED_GREEN,POSITION_OFF);
                switch( led ){
                    case UV_ON:     // turn on UV led
                        pwm_set(IR, 0);
                        pwm_set(WHITE, 0);
                        pwm_set(UV, pwmValue);

                        uart_txChar(2, UV_ON);
                        OUTPUT_SET(LED_GREEN,POSITION_ON);
                        uart_txChar(2, pwmValue);
                        OUTPUT_SET(LED_GREEN,POSITION_OFF);
                        valid = 1;
                        break;
                    case IR_ON:     // turn on IR led
                        pwm_set(UV, 0);
                        pwm_set(WHITE, 0);
                        pwm_set(IR, pwmValue);

                        uart_txChar(2, IR_ON);
                        OUTPUT_SET(LED_GREEN,POSITION_ON);
                        uart_txChar(2, pwmValue);
                        OUTPUT_SET(LED_GREEN,POSITION_OFF);
                        valid = 1;
                        break;
                    case WHITE_ON:  // turn on WHITE led
                        pwm_set(UV, 0);
                        pwm_set(IR, 0);
                        pwm_set(WHITE, pwmValue);

                        uart_txChar(2, WHITE_ON);
                        OUTPUT_SET(LED_GREEN,POSITION_ON);
                        uart_txChar(2, pwmValue);
                        OUTPUT_SET(LED_GREEN,POSITION_OFF);
                        valid = 1;
                        break;
                    case OFF:       // turn off leds
                        pwm_set(UV, 0);
                        pwm_set(IR, 0);
                        pwm_set(WHITE, 0);

                        uart_txChar(2, OFF);
                        OUTPUT_SET(LED_GREEN,POSITION_ON);
                        uart_txChar(2, pwmValue);
                        OUTPUT_SET(LED_GREEN,POSITION_OFF);
                        valid = 1;
                        break;
                    default:
                        // No valid command
                        uart_txChar(1,'e');
                        OUTPUT_SET(LED_GREEN,POSITION_ON);
                        uart_txChar(1,'r');
                        OUTPUT_SET(LED_GREEN,POSITION_OFF);
                        valid = 0;
                        break;
                }
            }while(valid == 0);
            valid = 0;

            // Master waits for the forwarded command to return at UART2
            // and answers at UART1
            timeout = 0;
            do{
                // wait for data to receive at UART2
                error = uart_rxChar(2, &led);
                timeout++;
            }while(error && timeout<10);
            OUTPUT_SET(LED_GREEN,POSITION_ON);
            uart_rxChar(2, &pwmValue);
            pwmValue &= 0x7F;
            OUTPUT_SET(LED_GREEN,POSITION_OFF);
            switch( led ){
                case UV_ON:
                    uart_txChar(1,UV_ON);
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(1, pwmValue);
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
                case IR_ON:
                    uart_txChar(1,IR_ON);
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(1, pwmValue);
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
                case WHITE_ON:
                    uart_txChar(1,WHITE_ON);
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(1, pwmValue);
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
                case OFF:
                    uart_txChar(1,OFF);
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(1, pwmValue);
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
                default:
                    // error occured
                    uart_txChar(1,'e');
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(1,'r');
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
            }

        } else {

            // Slave listens at UART2 and forwards the command at UART2
            do{
                // wait for data to receive at UART2
                error = uart_rxChar(2, &led);
            }while(error);
            OUTPUT_SET(LED_GREEN,POSITION_ON);
            uart_rxChar(2, &pwmValue);
            pwmValue &= 0x7F;
            OUTPUT_SET(LED_GREEN,POSITION_OFF);
            switch( led ){
                case UV_ON:    // turn on UV led
                    pwm_set(IR, 0);
                    pwm_set(WHITE, 0);
                    pwm_set(UV, pwmValue);

                    uart_txChar(2, UV_ON);
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(2, pwmValue);
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
                case IR_ON:    // turn on IR led
                    pwm_set(UV, 0);
                    pwm_set(WHITE, 0);
                    pwm_set(IR, pwmValue);

                    uart_txChar(2, IR_ON);
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(2, pwmValue);
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
                case WHITE_ON:  // turn on WHITE led
                    pwm_set(UV, 0);
                    pwm_set(IR, 0);
                    pwm_set(WHITE, pwmValue);

                    uart_txChar(2, WHITE_ON);
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(2, pwmValue);
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
                case OFF:       // turn off leds
                    pwm_set(UV, 0);
                    pwm_set(IR, 0);
                    pwm_set(WHITE, 0);

                    uart_txChar(2, OFF);
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(2, pwmValue);
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
                default:
                    // error occured
                    uart_txChar(2,'e');
                    OUTPUT_SET(LED_GREEN,POSITION_ON);
                    uart_txChar(2,'r');
                    OUTPUT_SET(LED_GREEN,POSITION_OFF);
                    break;
            }
        }
    }
}

void gpio_init(void)
{
    // activate port-clocks
    SIM->SCGC5 =    SIM_SCGC5_PORTA_MASK |
                    SIM_SCGC5_PORTB_MASK |
                    SIM_SCGC5_PORTC_MASK |
                    SIM_SCGC5_PORTD_MASK |
                    SIM_SCGC5_PORTE_MASK;

    // init outputs of LED's
    INIT_OUTPUT(LED_YELLOW);
    OUTPUT_SET(LED_YELLOW, POSITION_OFF);
    INIT_OUTPUT(LED_RED);
    OUTPUT_SET(LED_RED, POSITION_OFF);
    INIT_OUTPUT(LED_GREEN);
    OUTPUT_SET(LED_GREEN, POSITION_OFF);

    // init inputs of master/slave-code
    INIT_INPUT_PULLUP(CODE_0);
    INIT_INPUT_PULLUP(CODE_1);
    INIT_INPUT_PULLUP(CODE_2);
    INIT_INPUT_PULLUP(CODE_3);
}

void delay(void)
{
    volatile unsigned int i,j;

    for (i = 0U; i < 5000U; i++) {
        for (j = 0U; j < 1000U; j++) {
            __asm__("nop");
        }
    }
}
