/* Using BitNeuMCU for inference of 16x16 MNIST images on a CH32V003 */

#include "ch32fun.h"

// Latest version of CH32FUN seems to have more overhead. Hence, only three test patterns can be included.
// Declare processfclayer an SRAM based function for speedup
void processfclayer(int8_t *,  const uint32_t *, int32_t, uint32_t, uint32_t, int32_t *) __attribute__((section(".srodata"))) __attribute__((used));
int32_t *processmaxpool22(int32_t *activations, uint32_t xy_input, int32_t *output) __attribute__((section(".srodata"))) __attribute__((used));
int32_t* processconv33ReLU(int32_t *activations, const int8_t *weights, uint32_t xy_input, uint32_t  n_shift , int32_t *output) __attribute__((section(".srodata"))) __attribute__((used));	

// #include "BitNetMCU_model_1k.h"
// #include "BitNetMCU_model_12k.h"
// #include "BitNetMCU_model_12k_FP130.h"
// #include "BitNetMCU_model_cnn_48.h"
// #include "BitNetMCU_model_cnn_32.h"
// #include "BitNetMCU_model_cnn_16.h"
#include "BitNetMCU_model_cnn_64.h"
#include "../BitNetMCU_inference.c"
#include <stdio.h>

const int8_t input_data_0[256] = {-22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, 11.0, 64.0, 30.0, 6.0, -14.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, 28.0, 124.0, 127.0, 115.0, 66.0, -3.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -12.0, 18.0, 58.0, 97.0, 124.0, 70.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -16.0, 47.0, 100.0, -11.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -21.0, 44.0, 104.0, -11.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -16.0, 68.0, 106.0, -12.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -13.0, 77.0, 99.0, -18.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -13.0, 77.0, 96.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -13.0, 77.0, 96.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -13.0, 77.0, 96.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -17.0, 62.0, 97.0, -20.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, 18.0, 71.0, -14.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -22.0, -20.0, -16.0, -21.0, -22.0, -22.0, -22.0, -22.0, -22.0};
const uint8_t label_0 = 7;
const int8_t input_data_1[256] = {-20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -4.0, 69.0, 6.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 5.0, 106.0, 42.0, -18.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 28.0, 119.0, 50.0, -17.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -14.0, 64.0, 125.0, 19.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 3.0, 99.0, 121.0, 13.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -19.0, 33.0, 120.0, 100.0, -7.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -16.0, 71.0, 126.0, 65.0, -17.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 6.0, 106.0, 112.0, 13.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 47.0, 125.0, 100.0, -3.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 54.0, 127.0, 68.0, -19.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 53.0, 119.0, 43.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 16.0, 59.0, -3.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0};
const uint8_t label_1 = 1;
const int8_t input_data_2[256] = {-21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -16.0, 11.0, 49.0, 48.0, 0.0, -20.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -20.0, -5.0, 41.0, 80.0, 62.0, 56.0, 70.0, 0.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -18.0, 10.0, 76.0, 58.0, 3.0, -18.0, -14.0, 70.0, 24.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -19.0, 28.0, 94.0, 29.0, -15.0, -21.0, -21.0, 1.0, 99.0, 47.0, -20.0, -21.0, -21.0, -21.0, -21.0, -21.0, 7.0, 87.0, 29.0, -19.0, -21.0, -20.0, -9.0, 65.0, 90.0, 7.0, -21.0, -21.0, -21.0, -21.0, -21.0, -19.0, 55.0, 67.0, -14.0, -21.0, -20.0, -4.0, 77.0, 118.0, 30.0, -19.0, -21.0, -21.0, -21.0, -21.0, -21.0, -17.0, 68.0, 33.0, -12.0, -0.0, 29.0, 69.0, 127.0, 72.0, -12.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -18.0, 55.0, 91.0, 84.0, 75.0, 38.0, 51.0, 111.0, 8.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -11.0, 16.0, 14.0, -8.0, -13.0, 62.0, 65.0, -18.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -3.0, 84.0, 18.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, 14.0, 68.0, -13.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, 8.0, 39.0, -17.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -20.0, -18.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0};
const uint8_t label_2 = 9;
// const int8_t input_data_3[256] = {-20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -13.0, -15.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -6.0, 41.0, 78.0, 38.0, -18.0, -20.0, -20.0, -20.0, -20.0, -17.0, -17.0, -20.0, -20.0, -20.0, -20.0, -11.0, 67.0, 109.0, 63.0, 6.0, -20.0, -20.0, -20.0, -20.0, -8.0, 48.0, 50.0, -8.0, -20.0, -20.0, -20.0, 2.0, 108.0, 65.0, -14.0, -20.0, -20.0, -20.0, -20.0, -12.0, 59.0, 114.0, 89.0, 4.0, -20.0, -20.0, -20.0, 10.0, 114.0, 27.0, -20.0, -20.0, -20.0, -20.0, -20.0, 36.0, 122.0, 65.0, -14.0, -20.0, -20.0, -20.0, -20.0, -2.0, 96.0, 55.0, -13.0, -20.0, -20.0, -20.0, -12.0, 89.0, 114.0, 16.0, -20.0, -20.0, -20.0, -20.0, -20.0, -17.0, 43.0, 100.0, 46.0, -5.0, -15.0, -18.0, 6.0, 115.0, 84.0, -9.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -14.0, 45.0, 115.0, 100.0, 78.0, 50.0, 66.0, 127.0, 53.0, -17.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -13.0, 28.0, 76.0, 91.0, 104.0, 127.0, 122.0, 28.0, -18.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -19.0, -16.0, -14.0, -1.0, 71.0, 114.0, 8.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, 19.0, 112.0, 39.0, -13.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -11.0, 70.0, 89.0, 19.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -18.0, -6.0, -18.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0};
// const uint8_t label_3 = 4;

#ifdef MODEL_CNNMNIST

uint32_t BitMnistInference(const int8_t *input) {
    int32_t layer_out[256];  // has to hold 16x16 image
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

uint32_t BitMnistInference(const int8_t *input) {
    int32_t layer_out[MAX_N_ACTIVATIONS];
    int8_t layer_in[MAX_N_ACTIVATIONS];
	int32_t prediction;

    processfclayer((int8_t*)input, L1_weights, L1_bitperweight, L1_incoming_weights, L1_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L1_outgoing_weights);

    processfclayer(layer_in, L2_weights, L2_bitperweight, L2_incoming_weights,  L2_outgoing_weights, layer_out);
    ReLUNorm(layer_out, layer_in, L2_outgoing_weights);

    processfclayer(layer_in, L3_weights, L3_bitperweight, L3_incoming_weights,  L3_outgoing_weights, layer_out);
    prediction=ReLUNorm(layer_out, layer_in, L3_outgoing_weights);

#if NUM_LAYERS == 4
    processfclayer(layer_in, L4_weights, L4_bitperweight, L4_incoming_weights,  L4_outgoing_weights, layer_out);
    prediction=ReLUNorm(layer_out, layer_in, L4_outgoing_weights);
#endif

	return prediction;
}

#endif

void TestSample(const int8_t *input, const uint8_t label, const uint8_t sample) {
	volatile int32_t startticks, endticks;
    int32_t prediction;

	startticks = SysTick->CNT;
	prediction = BitMnistInference(input);
	endticks = SysTick->CNT;

	printf( "Inference of Sample %d\tPrediction: %ld\tLabel: %d\tTiming: %lu clock cycles\n", sample, prediction, label, endticks-startticks);	
}


int main()
{
	SystemInit();

	while(1)
	{
		printf("Starting MNIST inference...\n");
		TestSample(input_data_0, label_0,1);	
		TestSample(input_data_1, label_1,2);	
		TestSample(input_data_2, label_2,3);	
		// BitMnistInference(input_data_3, label_3,4);	
		Delay_Ms(1000);
	}
}

