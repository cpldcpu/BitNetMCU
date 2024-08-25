# 🚀 BitNetMCU Inference Engine Demo for CH32V003 MCU

This folder contains a demo that implements the BitNetMCU inference engine on an actual CH32V003 MCU with 16kb flash and 2kb ram. This example is to be used with the [ch32v003fun SDK](https://github.com/cnlohr/ch32v003fun).

## 📁 File Descriptions

- [`Makefile`](Makefile): The default makefile assumes that you clone ch32v003fun as a submodule. Alternatively, you can change the variable `CH32V003FUN` in the makefile to point to an alternative installation location. This Makefile will compile the demo and flash it to the CH32V003 MCU. 

- [`funconfig.h`](funconfig.h): Configuration of main clock speed, and SysTick clock source for the CH32V003 MCU.


- [`BitNetMCU_model_12k.h`](BitNetMCU_model_12k.h): A model optimized for high accuracy, achieving 99.02% test accuracy on MNIST. 4 Bit weights, 64/64/64 hidden layer width. 13.66 ms inference time.
  
- [`BitNetMCU_model_12k_FP130.h`](BitNetMCU_model_12k_FP130.h): A model optimizing for both accuracy and inference speed by using FP1.3.0 encoding. Achieving 98.86% test accuracy on MNIST. 4 Bit weights, 64/64/64 hidden layer width. 10.17 ms inference time.

- [`BitNetMCU_model_1k.h`](BitNetMCU_model_1k.h): A size optimized model with a memory footprint of only 1 kbyte, achieving 94.22% accuracy on MNIST. 2 Bit weights, 16/16/16 hidden layer width. Activate it by modifying `BitNetMCUdemo.c` 1.88 ms inference time.

- [`BitNetMCUdemo.c`](BitNetMCUdemo.c): This is the main C file for the demo. It includes the BitNetMCU inference engine and model from the main folder. It will perform inference on four included test images and output the results to the monitoring console. 

## 🛠️ Usage  

```
  make flash
  make monitor
```
Example output

![Example output on Monitor](console.png)