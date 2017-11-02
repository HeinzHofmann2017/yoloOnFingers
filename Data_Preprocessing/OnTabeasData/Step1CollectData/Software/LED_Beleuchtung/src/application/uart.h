/**************************************************
    Projekt:        Led Illumination
    File:           uart.h
    Date:           30.03.2017
    Author:         tmendez

    Description:    UART Driver for MKL26Z4

**************************************************/

#ifndef UART_H_
#define UART_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialises the UART
*  param:   nr:         UART to use ( 1 | 2 )
*           baudrate:   Baudrate to use
*/
void uart_init(uint8_t nr, unsigned int baudrate);

/* Transmits a single character over the UART
*  param:   nr:     UART to use ( 1 | 2 )
*           ch:     character to send
*/
void uart_txChar(uint8_t nr, uint8_t ch);

/* Receives a single character over the UART
*  param:   nr:     UART to use ( 1 | 2 )
*           ch:     buffer so save received character
*  return:  value >0 if not successful, 0 otherwise
*/
uint8_t uart_rxChar(uint8_t nr, uint8_t* ch);

#ifdef __cplusplus
}
#endif
#endif /* UART_H_ */
