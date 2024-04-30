# BitNetMCU: High Accuracy Low-Bit Quantized Neural Networks on a low-end Microcontroller

**BitNetMCU** is a project focused on the training and inference of low-bit quantized neural networks, specifically designed to run efficiently on low-end RISC-V microcontrollers like the CH32V003. Quantization aware training (QAT) and finetuning of model structure and inference code allowed *surpassing 99% Test accuracy on a 16x16 MNIST dataset without using multiplication instructions and in only 2kb of RAM and 16kb of Flash*.

**You can find a detailed report on the project in the `docs/` directory [here](docs/documentation.md).**

<div align="center">
    <img src="docs/header.png" width="95%">
</div>

## Project Structure

```
BitNetMCU/
│
├── docs/                      # Report
├── mcu/                       # MCU specific code for CH32V003
├── modeldata/                 # Pre-trained models
│
├── BitNetMCU.py               # Pytorch model and QAT classes
├── BitNetMCU_inference.c      # C code for inference 
├── BitNetMCU_inference.h      # Header file for C inference code
├── BitNetMCU_MNIST_test.c     # Test script for MNIST dataset
├── BitNetMCU_MNIST_test_data.h# MNIST test data in header format (generated)
├── BitNetMCU_model.h          # Model data in C header format (generated)
├── exportquant.py             # Script to convert trained model to quantized format
├── test_inference.py          # Script to test C implementation of inference
├── training.py                # Training script for the neural network
└── trainingparameters.yaml    # Configuration file for training parameters
```

## Training Pipeline

The data pipeline is split into several Python scripts for flexibility:

1. **Configuration**: Modify `trainingparameters.yaml` to set all hyperparameters for training the model.

2. **Training the Model**: The `training.py` script is used to train the model and store it as a `.pth` file in the `modeldata/` folder. The model weights are still in float format at this stage, as they are quantized on-the-fly during training.

2. **Exporting the Quantized Model**: The `exportquant.py` script is used to convert the model into a quantized format. The quantized model is exported to the C header file `BitNetMCU_model.h`.

3. **Optional: Testing the C-Model**: Compile and execute `BitNetMCU_MNIST_test.c` to test inference of ten digits. The model data is included from `BitNetMCU_MNIST_test_data.h`, and the test data is included from the `BitNetMCU_MNIST_test_data.h` file. 

4. **Optional: Verification C vs Python Model on full dataset**: The inference code, along with the model data, is compiled into a DLL. The `test-inference.py` script calls the DLL and compares the results with the original Python model. This allows for an accurate comparison to the entire MNIST test data set of 10,000 images.

5. **Optional: Testing inference on the MCU**: follow the instructions in  `mcu/readme.md`. Porting to architectures other than CH32V003 is straighfoward and the files in the `mcu` directory can serve as a reference

