# 🚀 BitNetMCU Inference Engine Demo for CH32V003 MCU

This folder contains a demo that implements the BitNetMCU inference engine on an actual CH32V003 MCU with 16kb flash and 2kb ram. This example is to be used with the [ch32v003fun SDK](https://github.com/cnlohr/ch32v003fun).

## 📁 File Descriptions

- [`Makefile`](Makefile): Make sure to update the `CH32V003FUN` variable to point to the location of the ch32v003fun SDK on your system. This Makefile will compile the demo and flash it to the CH32V003 MCU. 

- [`funconfig.h`](funconfig.h): Configuration of main clock speed, and SysTick clock source for the CH32V003 MCU.

- [`BitNetMCUdemo.c`](BitNetMCUdemo.c): This is the main C file for the demo. It includes the BitNetMCU inference engine and model from the main folder. It will perform inference on four included test images and output the results to the monitoring console. 

## 🛠️ Usage  

```
  make flash
  make monitor
```
Example output

![Example output on Monitor](console.png)