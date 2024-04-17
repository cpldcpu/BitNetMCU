#include <stdint.h>
#include <stdio.h>

#include "BitNetMCU_model.h"
#include "BitNetMCU_inference.c"

/**
 * @file Bitnet_inference_lib.c
 * @brief DLL wrapper for the BitMnist model.
 *  build in 64 bit visual studio:
 *    cl /LD BitNetMCU_MNIST_dll.c /MD /FeBitnet_inf.dll /link /MACHINE:X64
 *@param input The input data for the inference.
 * @return The result of the inference.
 */

#ifdef _DLL
__declspec(dllexport) uint32_t Inference(int8_t *input) {
    return BitMnistInference(input);
}
#endif

void printactivations(uint8_t *activations, int32_t n_activations) 
{
    for (int i = 0; i < n_activations; i++) {
        printf("%d, ", activations[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
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