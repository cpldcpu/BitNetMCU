/*
    BitNetMCU inference functions
    @cpldcpu April 2024

    Performs inference on fully connected layer on a very resource contrained MCU.
    1,2,4 bit weights are supported.
    
*/

#include <stdint.h>
#include <stdio.h>
#include "BitNetMCU_inference.h"

/**
 * @brief Applies a ReLU activation function to an array of integers and normalizes the result to 8-bit integers.
 * 
 * @param input Pointer to the input array of 32-bit integers.
 * @param output Pointer to the output array of 8-bit integers.
 * @param n_input The number of elements in the input array.
 * @return The position of maximum value found in the input array before applying the ReLU activation.
 */

uint32_t ReLUNorm(int32_t *input, int8_t *output, uint32_t n_input) {
    int32_t max_val = -INT32_MAX;
    int32_t max_pos = 255;
    uint32_t scale; 
    uint32_t shift; 
    int32_t rounding;
    int32_t tmp;

    // Find the maximum value in the input array
    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_pos = i;
        }
    }

    // Normalization 
    // Dynamic shift according to max value in the input array
    scale=max_val>>7;  // define max range, all bits above 7 will be shifted down
    shift=0;

    while (scale>0) {
        shift++;
        scale>>=1;
    }

    // impact of rounding is almost negligible (+0.03% in eval accuracy)
    // But rounding affects mismatch to python inference engine
    rounding   = 1 << (shift - 1);  

    // Apply ReLU activation and normalize to 8-bit
    for (uint32_t i = 0; i < n_input; i++) {
        // Apply ReLU activation
        if (input[i] < 0) {
            output[i] = 0;
        } else {
            tmp=(input[i] + rounding) >> shift;  

            // clipping needed to catch overflow from rounding
            if (tmp > 127) { 
                output[i] = 127;
            } else {
            output[i] = tmp;
            }
        }    
    }
    return max_pos;
}

/**
 * @brief Processes a fully connected layer in a neural network.
 *
 * This function processes a fully connected layer in a neural network by performing
 * the dot product of the input activations and weights, and stores the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.
 * @param n_input The number of input neurons.
 * @param n_output The number of output neurons.
 * @param output Pointer to the output array where the result of the layer is stored.
 */

void processfclayer( int8_t *activations,  const uint32_t *weights, int32_t bits_per_weight, uint32_t n_input, uint32_t n_output, int32_t *output) 
{
   const uint32_t *weightidx = weights;
    
    for (uint32_t i = 0; i < n_output; i++) {
        int8_t *activations_idx = activations;
        int32_t sum = 0;

        if (bits_per_weight == 1) {
            for (uint32_t k = 0; k < n_input; k+=32) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 32; j++) {
                    int32_t in=*activations_idx++;
                    sum += (weightChunk & 0x80000000) ? in : -in;  // Note that sign is flipped for Binary quant (bit set equals positive)
                    weightChunk <<= 1;
                }
            }
        } else if (bits_per_weight == 2 ) {
            for (uint32_t k = 0; k < n_input; k+=16) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 16; j++) {
                    int32_t in=*activations_idx++;
                    int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                    sum += tmpsum;                                  // sign*in*1
                    if (weightChunk & 0x40000000) sum += tmpsum<<1; // sign*in*2
                    weightChunk <<= 2;
                }
            }
        // Muliplier-less inference for RB32EC
#if defined(__riscv) && !defined(__riscv_mul)          
        } else if (bits_per_weight == 4 ) { 
            for (uint32_t k = 0; k < n_input; k+=8) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    if (in != 0) { // Skip zero activations to speed up inference in layers after first layer
                        int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                        sum += tmpsum;                                  // sign*in*1
                        if (weightChunk & 0x10000000) sum += tmpsum<<1; // sign*in*2
                        if (weightChunk & 0x20000000) sum += tmpsum<<2; // sign*in*4
                        if (weightChunk & 0x40000000) sum += tmpsum<<3; // sign*in*8
                    }
                    weightChunk <<= 4;
                }
            }
#else
        } else if (bits_per_weight == 4 ) { 
            for (uint32_t k = 0; k < n_input; k+=8) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    if (in != 0) { // Skip zero activations to speed up inference in layers after first layer
                        int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                        sum += tmpsum * ((weightChunk>>(32-4))&7);                                  // sign*in*1
                    }
                    weightChunk <<= 4;
                }
            }
#endif
        } else if (bits_per_weight == 8 + 4 ) {   // 4 bit twos-complement
            for (uint32_t k = 0; k < n_input; k+=8) {
                int32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    int32_t weight = (weightChunk) >> (32-4); // extend sign, cut off lower bits
                    sum += in*weight;                                  
                    weightChunk <<= 4;
                }
            }
        }  else if (bits_per_weight == 16 + 4 ) {  // 4 bit shift
            for (uint32_t k = 0; k < n_input; k+=8) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    int32_t tmpsum;
                    
                    tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                    sum += tmpsum << ((weightChunk >> 28) & 7); // sign*in*2^log                       
                    weightChunk <<= 4;
                }
            }
        }   // else printf("Error: unsupported weight bit width %d\n", bits_per_weight);
       
        output[i] = sum;
        // printf("%d,", output[i]);  
    }
}