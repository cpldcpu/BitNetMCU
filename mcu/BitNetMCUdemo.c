#include "../BitNetMCU_inference.c"

#include <ch32v003fun.h>
#include <stdio.h>
#include <string.h>
// #include "BitNetMCU_model_1k.h"
#include "BitNetMCU_model_12k.h"
// #include "BitNetMCU_model_12k_FP130.h"

#define LED_PIN 4
#define UART_RX_PIN 6
#define ARRAY_SIZE 256
#define BUFFER_SIZE 800

int8_t received_integers[ARRAY_SIZE] = {0};
volatile int array_index = 0;
volatile int8_t input_received = 0; // Flag to indicate input received
volatile char buffer[BUFFER_SIZE];  // Buffer to store received data
volatile int buf_index = 0;         // Index for the buffer

// Simple implementation of atoi
int8_t simple_atoi(const char * str) {
    int8_t result = 0;
    int8_t sign = 1;

    // Handle negative numbers
    if (*str == '-') {
        sign = -1;
        str++;
    }

    // Convert string to integer
    while (*str) {
        result = result * 10 + (*str - '0');
        str++;
    }

    return sign * result;
}

void USART1_IRQHandler(void) __attribute__((interrupt));
void USART1_IRQHandler(void) {
    if (USART1->STATR & USART_STATR_RXNE) {
        // Read the received data
        int received_byte = USART1->DATAR;

        if (received_byte == '\n' || received_byte == '\r') {
            buffer[buf_index] = '\0'; // Null-terminate the string
            char * token = (char *)buffer;
            while (*token != '\0') {
                // Find the next space character
                char * next_space = strchr(token, ' ');

                // Null-terminate the token
                if (next_space != NULL) {
                    *next_space = '\0';
                }

                // Convert the token to integer and store it in the array
                received_integers[array_index++] = simple_atoi(token);
                // printf("%d", received_integers[array_index-1]);
                //  Move to the next token
                token = next_space + 1;

                // Wrap around if the index exceeds the array size
                if (array_index >= ARRAY_SIZE) {
                    array_index = 0;
                }
            }
            buf_index = 0;      // Reset buffer index
            input_received = 1; // Set the flag to indicate input received
        }
        else {
            buffer[buf_index++] = received_byte;
        }
    }
}

void setup_uart(void) {
    RCC->APB2PCENR |= RCC_APB2Periph_USART1; // Enable USART1 clock
    RCC->APB2PCENR |= RCC_APB2Periph_GPIOD;  // Enable GPIO D clock

    // Configure PD6 as USART1_RX
    GPIOD->CFGLR &= ~(0xF << (UART_RX_PIN * 4));
    GPIOD->CFGLR |= (0x8 << (UART_RX_PIN * 4)); // Input with pull-up / pull-down

    // Configure USART1
    // Baud rate 115200
    USART1->BRR = 0x1A1;                 // 115200 baud rate
    USART1->CTLR1 |= USART_CTLR1_RE;     // Enable receiver
    USART1->CTLR1 |= USART_CTLR1_RXNEIE; // Enable RX interrupt
    USART1->CTLR1 |= USART_CTLR1_UE;     // Enable USART

    NVIC_EnableIRQ(USART1_IRQn); // Enable USART1 interrupt in NVIC
}

void setup_gpio(void) {
    RCC->APB2PCENR |= RCC_APB2Periph_GPIOD; // Enable GPIO D clock

    // Configure PD4 as output
    GPIOD->CFGLR &= ~(0xF << (LED_PIN * 4));
    GPIOD->CFGLR |= (0x1 << (LED_PIN * 4)); // Output push-pull, max speed 10 MHz
}

void display_received_integers(void) {
    printf("Received integers: [");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (i != 0) {
            printf(", ");
        }
        printf("%d", received_integers[i]);
    }
    printf("]\n");
}

/* Using BitNeuMCU for inference of 16x16 MNIST images on a CH32V003 */

// Declare processfclayer an SRAM based function for speedup
void processfclayer(int8_t *, const uint32_t *, int32_t, uint32_t, uint32_t, int32_t *)
    __attribute__((section(".srodata"))) __attribute__((used));

void BitMnistInference(const int8_t * input, const uint8_t sample) {
    int32_t layer_out[MAX_N_ACTIVATIONS];
    int8_t layer_in[MAX_N_ACTIVATIONS];
    int32_t prediction;
    uint32_t startticks, endticks;

    startticks = SysTick->CNT;
    processfclayer((int8_t *)input, L1_weights, L1_bitperweight, L1_incoming_weights,
        L1_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L1_outgoing_weights);

    processfclayer(layer_in, L2_weights, L2_bitperweight, L2_incoming_weights,
        L2_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L2_outgoing_weights);

    processfclayer(layer_in, L3_weights, L3_bitperweight, L3_incoming_weights,
        L3_outgoing_weights, layer_out);
    prediction = ReLUNorm(layer_out, layer_in, L3_outgoing_weights);

#if NUM_LAYERS == 4
    processfclayer(layer_in, L4_weights, L4_bitperweight, L4_incoming_weights,
        L4_outgoing_weights, layer_out);
    prediction = ReLUNorm(layer_out, layer_in, L4_outgoing_weights);
#endif

    endticks = SysTick->CNT;

    printf("Inference of Sample %d\t Prediction: %ld\t Timing: %lu clock cycles\n",
        sample, prediction, endticks - startticks);
}

int main() {
    SystemInit();
    SysTick->CTLR = 5;  // Use HCLK as time base -> configured in funconfig.h
    setup_uart();
    setup_gpio();
    printf("MCU initialized\n");
    while (1) {
        // Toggle the LED
        GPIOD->OUTDR ^= (1 << LED_PIN);
        Delay_Ms(250);

        // Check if input was received
        if (input_received) {
            input_received = 0;          // Reset the flag
            display_received_integers(); // Display the received integers array
            BitMnistInference(received_integers, 1);
        }

        // Print a message in the main while loop
        printf("Main loop is running...\n");
        Delay_Ms(250);
    }
    return 0;
}
