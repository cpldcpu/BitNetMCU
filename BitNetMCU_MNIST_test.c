#include <stdint.h>
#include <stdio.h>

#include "BitNetMCU_model.h"
#include "BitNetMCU_inference.c"
#include "BitNetMCU_MNIST_test_data.h"
/**
 * Performs inference on the MNIST dataset using the BitNetMCU model.
 *
 * @param input The input data for the inference, a 16x16 array of int8_t.
 * @return The predicted digit.
 */

uint32_t BitMnistInference(int8_t*);

void main(void) {
    uint32_t output[10];
    uint8_t predicted_label;
    predicted_label = BitMnistInference(input_data_0);
    printf("label: %d predicted: %d\n", label_0, predicted_label);
    predicted_label = BitMnistInference(input_data_1);
    printf("label: %d predicted: %d\n", label_1, predicted_label);
    predicted_label = BitMnistInference(input_data_2);
    printf("label: %d predicted: %d\n", label_2, predicted_label);
    predicted_label = BitMnistInference(input_data_3);
    printf("label: %d predicted: %d\n", label_3, predicted_label);
    predicted_label = BitMnistInference(input_data_4);
    printf("label: %d predicted: %d\n", label_4, predicted_label);
    predicted_label = BitMnistInference(input_data_5);
    printf("label: %d predicted: %d\n", label_5, predicted_label);
    predicted_label = BitMnistInference(input_data_6);
    printf("label: %d predicted: %d\n", label_6, predicted_label);
    predicted_label = BitMnistInference(input_data_7);
    printf("label: %d predicted: %d\n", label_7, predicted_label);
    predicted_label = BitMnistInference(input_data_8);
    printf("label: %d predicted: %d\n", label_8, predicted_label);
    predicted_label = BitMnistInference(input_data_9);
    printf("label: %d predicted: %d\n", label_9, predicted_label);
}

/**
 * @brief Performs inference on the BitMnist model.
 *
 * @param input The input data for the inference.
 * @return The result of the inference.
 */
uint32_t BitMnistInference(int8_t *input) {
    int32_t layer_out[256];
    uint8_t layer_in[256];

    processfclayer(input, L1_weights, L1_bitperweight, L1_incoming_weights, L1_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L1_outgoing_weights);

    processfclayer(layer_in, L2_weights, L2_bitperweight, L2_incoming_weights,  L2_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L2_outgoing_weights);

    processfclayer(layer_in, L3_weights, L3_bitperweight, L3_incoming_weights,  L3_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L3_outgoing_weights);

    processfclayer(layer_in, L4_weights, L4_bitperweight, L4_incoming_weights,  L4_outgoing_weights, layer_out);
    return ReLUNorm(layer_out, layer_in, L4_outgoing_weights);
}