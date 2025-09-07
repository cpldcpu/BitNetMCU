# 🚀 BitNetMCU Inference Engine Demo for CH32V003 (and other) MCU

This folder contains a demo that implements the BitNetMCU inference engine on an actual CH32V003 MCU with 16kb flash and 2kb ram.  This example is to be used with the [ch32fun SDK](https://github.com/cnlohr/ch32fun).

## File Descriptions

- [`Makefile`](Makefile): The default makefile assumes that you clone ch32v003fun as a submodule. This Makefile will compile the demo and flash it to the MCU. Change this file to retarget other MCUs.

- [`funconfig.h`](funconfig.h): Configuration of main clock speed, and SysTick clock source.

- [`BitNetMCUdemo.c`](BitNetMCUdemo.c): This is the main C file for the demo. It includes the BitNetMCU inference engine and model from the main folder. It will perform inference on included test images and output the results to the monitoring console. 

## Models

Different models can be selected by including the respective header file in `BitNetMCUdemo.c`. The following model file are included. Execution timings are measured on a CH32V002 at 48MHz. 

| File Name | Configuration | CNN Width | Size (kB) | Test Accuracy | Cycles Avg. | Time (ms) |
|-----------|---------------|-------|-----------|---------------|-------------|-----------|
| `BitNetMCU_model_cnn_16small.h` | 16-wide CNN, small fc   | 16    | 3.2       | 98.92%        | 686,490     | 14.30     |
| `BitNetMCU_model_cnn_16.h` | 16-wide CNN   | 16    | 5.4       | 99.06%        | 785,123     | 16.36     |
| `BitNetMCU_model_cnn_32.h` | 32-wide CNN   | 32    | 7.3       | 99.28%        | 1,434,667   | 29.89     |
| `BitNetMCU_model_cnn_48.h` | 48-wide CNN   | 48    | 9.3       | 99.44%        | 2,083,568   | 43.41     |
| `BitNetMCU_model_cnn_64.h` | 64-wide CNN   | 64    | 11.0      | 99.55%        | 2,736,250   | 57.01     |
| `BitNetMCU_model_1k.h` | 1k 2Bitsym FC   | -     | 1.1       | 94.22%        | 99,783      | 2.08      |
| `BitNetMCU_model_12k.h` | 12k 4Bitsym FC  | -     | 12.3      | 99.02%        | 528,377     | 11.01     |
| `BitNetMCU_model_12k_FP130.h` | 12k FP130 FC  | -     | 12.3      | 98.86%        | 481,624     | 10.03     |

Take a look at the documentation for more details on the model architecture and trade offs: [Documentation](../docs/documentation.md)

## Usage  

```
  make flash
  make monitor
```
Example output

![Example output on Monitor](console.png)
