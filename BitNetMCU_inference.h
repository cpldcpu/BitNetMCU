#ifndef BITNETMCU_INFERENCE_H
#define BITNETMCU_INFERENCE_H

#include <stdint.h>

/**
 * @brief Applies a ReLU activation function to an array of integers and normalizes the result to 8-bit integers.
 * 
 * @param input Pointer to the input array of 32-bit integers.
 * @param output Pointer to the output array of 8-bit integers.
 * @param n_input The number of elements in the input array.
 * @return The position of maximum value found in the input array before applying the ReLU activation.
 */

uint32_t ReLUNorm(int32_t *input, int8_t *output, uint32_t n_inpu);


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
void processfclayer(int8_t *input, const uint32_t *weights, int32_t bits_per_weight, uint32_t incoming_weights, uint32_t outgoing_weights, int32_t *output);


/**
 * @brief fused 3x3 conv2d and ReLU activation function
   
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param xy_input The number of input neurons.
 * @param n_shift The number of bits to shift the result of the convolution after summation, typically either 8+3=11 or 8+4=12.
 * @param output Pointer to the output array where the result of the layer is stored.
 * @return Pointer to the end of the output array.
 */

int32_t* processconv33ReLU(int32_t *activations, const int8_t *weights, uint32_t xy_input, uint32_t  n_shift , int32_t *output);


/**
 * @brief maxpool2d 2x2 function
 *
 * This function performs a 2x2 max pooling operation on a 2D array of input activations.
 * The function divides the input activations into 2x2 non-overlapping regions and selects the maximum value in each region.
 * *  
 * @param activations Pointer to the input activations of the layer.
 * @param xy_input The number of input neurons.
 * @param output Pointer to the output array where the result of the layer is stored.
 * @return Pointer to the end of the output array.
 */

int32_t *processmaxpool22(int32_t *activations, uint32_t xy_input, int32_t *output);



#endif // BITNETMCU_INFERENCE_H