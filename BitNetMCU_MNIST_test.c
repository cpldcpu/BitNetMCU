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


#ifdef MODEL_CNNMNIST

uint32_t BitMnistInference(int8_t *input) {
    int32_t layer_out[MAX_N_ACTIVATIONS];
    int8_t layer_in[MAX_N_ACTIVATIONS*4];

    /*
        Layer: L2 Conv2d bpw: 8 1 -> 64 groups:1 Kernel: 3x3 Incoming: 16x16 Outgoing: 14x14
        Layer: L4 Conv2d bpw: 8 64 -> 64 groups:64 Kernel: 3x3 Incoming: 14x14 Outgoing: 12x12
        Layer: L6 MaxPool2d Pool Size: 2 Incoming: 12x12 Outgoing: 6x6
        Layer: L7 Conv2d bpw: 8 64 -> 64 groups:64 Kernel: 3x3 Incoming: 6x6 Outgoing: 4x4
        Layer: L9 MaxPool2d Pool Size: 2 Incoming: 4x4 Outgoing: 2x2
        Layer: L11 Quantization type: <2bitsym>, Bits per weight: 2, Num. incoming: 256,  Num outgoing: 96
        Layer: L13 Quantization type: <4bitsym>, Bits per weight: 4, Num. incoming: 96,  Num outgoing: 64
        Layer: L15 Quantization type: <4bitsym>, Bits per weight: 4, Num. incoming: 64,  Num outgoing: 10
    */

    // Depthwise separable convolution with 32 bit activations and 8 bit weights
    int32_t *tmpbuf=(int32_t*)layer_out;
    int32_t *outputptr=(int32_t*)layer_in;
    for (uint32_t channel=0; channel < L7_out_channels; channel++) {

        for (uint32_t i=0; i < 16*16; i++) {
            tmpbuf[i]=input[i];
        }
        processconv33ReLU(tmpbuf,  L2_weights + 9*channel,  L2_incoming_x, 4, tmpbuf);
        processconv33ReLU(tmpbuf, L4_weights + 9*channel,  L4_incoming_x, 4, tmpbuf);
        processmaxpool22(tmpbuf, L6_incoming_x, tmpbuf);
        processconv33ReLU(tmpbuf, L7_weights + 9*channel,  L7_incoming_x, 4, tmpbuf);
         
        outputptr= processmaxpool22(tmpbuf, L9_incoming_x, outputptr);
    }

    // Normalization and conversion to 8-bit
    ReLUNorm((int32_t*)layer_in, layer_in, L7_out_channels * L9_outgoing_x * L9_outgoing_y);

    // Fully connected layers
    processfclayer(layer_in, L11_weights, L11_bitperweight, L11_incoming_weights,  L11_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L11_outgoing_weights);

    processfclayer(layer_in, L13_weights, L13_bitperweight, L13_incoming_weights,  L13_outgoing_weights, layer_out);    
    ReLUNorm(layer_out, layer_in, L13_outgoing_weights);

    processfclayer(layer_in, L15_weights, L15_bitperweight, L15_incoming_weights,  L15_outgoing_weights, layer_out);
    return ReLUNorm(layer_out, layer_in, L15_outgoing_weights);
}

#elif defined(MODEL_FCMNIST)

uint32_t BitMnistInference(int8_t *input) {
    int32_t layer_out[MAX_N_ACTIVATIONS];
    int8_t layer_in[MAX_N_ACTIVATIONS];
   
    processfclayer(input, L1_weights, L1_bitperweight, L1_incoming_weights, L1_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L1_outgoing_weights);
    
    // printf("L1 activations:  \n");
    // printactivations(layer_in, L1_outgoing_weights);

    processfclayer(layer_in, L2_weights, L2_bitperweight, L2_incoming_weights, L2_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L2_outgoing_weights);

    // printf("L2 activations:  \n");
    // printactivations(layer_in, L2_outgoing_weights);

    #ifdef L4_active
        processfclayer(layer_in, L3_weights, L3_bitperweight, L3_incoming_weights, L3_outgoing_weights, layer_out);
        ReLUNorm(layer_out, layer_in, L3_outgoing_weights);

        processfclayer(layer_in, L4_weights, L4_bitperweight, L4_incoming_weights, L4_outgoing_weights, layer_out);
        return ReLUNorm(layer_out, layer_in, L4_outgoing_weights);
    #else
        processfclayer(layer_in, L3_weights, L3_bitperweight, L3_incoming_weights, L3_outgoing_weights, layer_out);
        return ReLUNorm(layer_out, layer_in, L3_outgoing_weights);
    #endif
}
#else
    #error "No model defined"
#endif