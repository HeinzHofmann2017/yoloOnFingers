/**************************************************
    Projekt:        Led Illumination
    File:           uart.c
    Date:           30.03.2017
    Author:         tmendez

    Description:    UART Driver for MKL26Z4

**************************************************/

#include <stdint.h>

#include "MKL26Z4.h"
#include "uart.h"


#define UART_CLOCK  24000000
#define TIMEOUT_CNT 1000000

#if !defined(UART_CLOCK)
#error "UART_CLOCK not defined"
#endif


void uart_init(uint8_t nr, unsigned int baudrate) {

    if (nr == 1){   /* UART1 */

        // Enable clock to UART
        SIM->SCGC4 |= SIM_SCGC4_UART1_MASK;

        // Enable clock to port pins used by UART
        SIM->SCGC5 |= SIM_SCGC5_PORTE_MASK;     // UART1

        // Select Tx & Rx pins to use (not necessary for uart2)
        SIM->SOPT5 &= ~(SIM_SOPT5_UART1RXSRC_MASK|SIM_SOPT5_UART1TXSRC_MASK);

        // Set Tx & Rx Pin function
        PORTE->PCR[0] = PORT_PCR_MUX(3);
        PORTE->PCR[1] = PORT_PCR_MUX(3);

        // Disable UART before changing registers
        UART1->C2 &= ~(UART_C2_TE_MASK | UART_C2_RE_MASK);

        // disable interrupts
        UART1->BDH &= ~UART_BDH_LBKDIE_MASK;
        UART1->BDH &= ~UART_BDH_RXEDGIE_MASK;
        UART1->C2 &= ~(UART_C2_TIE_MASK | UART_C2_TCIE_MASK | UART_C2_RIE_MASK | UART_C2_ILIE_MASK);

        // configure uart-frame
        UART1->BDH &= ~UART_BDH_SBNS_MASK;                  // 1 stop bit
        UART1->C1 |= UART_C1_M_MASK;                        // 8 data bits + 1 parity bit
        UART1->C1 |= (UART_C1_PE_MASK | UART_C1_PT_MASK);   // Odd-Parity

        // Calculate UART clock setting (5-bit fraction at right)
        uint32_t scaledBaudValue = (2*UART_CLOCK)/(baudrate);
        scaledBaudValue += 1<<4; // Round value
        // Set Baud rate register
        UART1->BDH = (UART1->BDH&~UART_BDH_SBR_MASK) | UART_BDH_SBR((scaledBaudValue>>(8+5)));
        UART1->BDL = UART_BDL_SBR(scaledBaudValue>>5);

        // Enable UART Tx & Rx
        UART1->C2 |= (UART_C2_TE_MASK | UART_C2_RE_MASK);

    } else if (nr == 2){   /* UART2 */

        // Enable clock to UART
        SIM->SCGC4 |= SIM_SCGC4_UART2_MASK;

        // Enable clock to port pins used by UART
        SIM->SCGC5 |= SIM_SCGC5_PORTD_MASK;     // UART2

        // Select Tx & Rx pins to use (not necessary for uart2)

        // Set Tx & Rx Pin function
        PORTD->PCR[2] = PORT_PCR_MUX(3);
        PORTD->PCR[3] = PORT_PCR_MUX(3);

        // Disable UART before changing registers
        UART2->C2 &= ~(UART_C2_TE_MASK | UART_C2_RE_MASK);

        // disable interrupts
        UART2->BDH &= ~UART_BDH_LBKDIE_MASK;
        UART2->BDH &= ~UART_BDH_RXEDGIE_MASK;
        UART2->C2 &= ~(UART_C2_TIE_MASK | UART_C2_TCIE_MASK | UART_C2_RIE_MASK | UART_C2_ILIE_MASK);

        // configure uart-frame
        UART2->BDH &= ~UART_BDH_SBNS_MASK;                  // 1 stop bit
        UART2->C1 |= UART_C1_M_MASK;                        // 8 data bits + 1 parity bit
        UART2->C1 |= (UART_C1_PE_MASK | UART_C1_PT_MASK);   // Odd-Parity

        // Calculate UART clock setting (5-bit fraction at right)
        uint32_t scaledBaudValue = (2*UART_CLOCK)/(baudrate);
        scaledBaudValue += 1<<4; // Round value
        // Set Baud rate register
        UART2->BDH = (UART2->BDH&~UART_BDH_SBR_MASK) | UART_BDH_SBR((scaledBaudValue>>(8+5)));
        UART2->BDL = UART_BDL_SBR(scaledBaudValue>>5);

        // Enable UART Tx & Rx
        UART2->C2 |= (UART_C2_TE_MASK | UART_C2_RE_MASK);
    }
}




void uart_txChar(uint8_t nr, uint8_t ch) {

    if (nr == 1) {   /* UART1 */

        while ((UART1->S1 & UART_S1_TDRE_MASK) == 0) {
            // Wait for Tx buffer empty
            __asm__("nop");
        }
        UART1->D = ch;

    } else if (nr == 2) {   /* UART2 */

        while ((UART2->S1 & UART_S1_TDRE_MASK) == 0) {
            // Wait for Tx buffer empty
            __asm__("nop");
        }
        UART2->D = ch;

    }

}



uint8_t uart_rxChar(uint8_t nr, uint8_t* ch) {
    uint32_t timeout = 0;
    uint8_t status;

    if (nr == 1){   /* UART1 */

        // Wait for Rx buffer full
        do {
            status = UART1->S1;
            // Clear & ignore pending errors
            if ((status & (UART_S1_FE_MASK|UART_S1_OR_MASK|UART_S1_PF_MASK|UART_S1_NF_MASK)) != 0) {
				// error occurred -> clear received data
                (void)UART1->D;
            }
            timeout++;
        } while (((status & UART_S1_RDRF_MASK) == 0) && (timeout < TIMEOUT_CNT));

        // get received data
        if(timeout < TIMEOUT_CNT){
            // received data successful
            *ch = UART1->D;
            return 0;
        } else {
            // timeout occurred
            *ch = 0x00;
            return 1;
        }

    } else if (nr == 2){   /* UART2 */

        // Wait for Rx buffer full
        do {
            status = UART2->S1;
            // Clear & ignore pending errors
            if ((status & (UART_S1_FE_MASK|UART_S1_OR_MASK|UART_S1_PF_MASK|UART_S1_NF_MASK)) != 0) {
				// error occurred -> clear received data
                (void)UART2->D;
            }
            timeout++;
        } while (((status & UART_S1_RDRF_MASK) == 0) && (timeout < TIMEOUT_CNT));

        // get received data
        if(timeout < TIMEOUT_CNT){
            // received data successful
            *ch = UART2->D;
            return 0;
        } else {
            // timeout occurred
            *ch = 0x00;
            return 1;
        }
    }

    return 1; // error occurred
}
