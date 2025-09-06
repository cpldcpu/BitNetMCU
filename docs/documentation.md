# BitNetMCU

**Surpassing 99% MNIST Test Accuracy with Low-Bit Quantized Neural Networks on a low-end RISC-V Microcontroller**- [BitNetMCU](#bitnetmcu)
- [Introduction and Motivation](#introduction-and-motivation)
  - [Background](#background)
- [Implementation of training code](#implementation-of-training-code)
- [Model Optimization](#model-optimization)
  - [Quantization Aware Training vs Post-Quantization](#quantization-aware-training-vs-post-quantization)
    - [Model Capacity vs Quantization scaling](#model-capacity-vs-quantization-scaling)
    - [Test Accuracy and Loss](#test-accuracy-and-loss)
  - [Optimizing training parameters](#optimizing-training-parameters)
    - [Learning rate and number of epochs](#learning-rate-and-number-of-epochs)
    - [Data Augmentation](#data-augmentation)
- [Architecture of the Inference Engine](#architecture-of-the-inference-engine)
  - [Implementation in Ansi-C](#implementation-in-ansi-c)
    - [fc-layer](#fc-layer)
    - [ShiftNorm / ReLU block](#shiftnorm--relu-block)
- [Putting it all together](#putting-it-all-together)
  - [Model Exporting](#model-exporting)
  - [Verification of the Ansi-C Inference Engine vs. Python](#verification-of-the-ansi-c-inference-engine-vs-python)
  - [Implementation on the CH32V003](#implementation-on-the-ch32v003)
- [Summary and Conclusions](#summary-and-conclusions)
- [Updates](#updates)
  - [May 20, 2024: Additional quantization schemes](#may-20-2024-additional-quantization-schemes)
    - [FP1.3.0 Quantization](#fp130-quantization)
    - [4-bit ones complement quantization](#4-bit-ones-complement-quantization)
  - [May 20, 2024: Quantization scaling](#may-20-2024-quantization-scaling)
  - [July 19, 2024: OCTAV Optimum Clipping](#july-19-2024-octav-optimum-clipping)
  - [July 26, 2024: NormalFloat4 (NF4) Quantization](#july-26-2024-normalfloat4-nf4-quantization)
  - [Aug 3rd, 2024: Stepped Learning Rate Schedule](#aug-3rd-2024-stepped-learning-rate-schedule)
  - [Aug 31st, 2025: Porting to CH32V002 and newest version of CH32fun](#aug-31st-2025-porting-to-ch32v002-and-newest-version-of-ch32fun)
    - [Execution Time Comparison between CH32V002 and CH3V003](#execution-time-comparison-between-ch32v002-and-ch3v003)
- [References](#references)


# Introduction and Motivation

Recently, there has been considerable hype about large language models (LLMs) with "1 Bit" or "1.58 Bit" [^1] weight quantization. The claim is that, by using Quantization Aware Training (QAT), LLMs can be trained with almost no loss of quality when using only binary or ternary encoding of weights. 

Interestingly, low bit quantization is also advantageous for inference on microcontrollers. The CH32V003 microcontroller gained some notoriety being extremely low cost for a 32 bit MCU (less than $0.15 in low volume), but is also notable for the RV32EC ISA, which supports only 16 registers and lacks a hardware multiplier. It also only has 16kb of flash and 2kb of ram.

The use of a few bits for each weight encoding means that less memory is required to store weights, and inference can be performed using only additions. Thus, the absence of a multiplication instruction is not an impediment to running inference on this microcontroller.

The challenge I set to myself: Develop a pipeline for training and inference of low-bit quantized neural networks to run on a CH32V003 RISC-V microcontroller. As is common, I will use the MNIST dataset for this project and try to achieve as high test accuracy as possible.

This document is quite lengthy and rather serves as a personal record of experiments.

## Background
 
Quantization of Deep Neural Networks (DNN) is not a novel concept, a review can be found in [^2]. Different approaches are usually distinguished by three characteristics:

**Pre- or post-training quantization** - The quantization of weights and activations can be done during training (QAT) or after training (post-quantization). QAT allows to consider the impact of quantization on the network already during training time. This comes at the expense of increased training time, complexity and loss of flexibility. QAT is most suitable for situations where the network is trained once, and inference takes place in a device with less computing power and memory ("edge device"). Post-quantization is more flexible and can be used to quantize a network to different bit-widths after training. 

**Quantization granularity of weights** - The number of bits used to encode the weights. Networks are typically trained with floating point weights, but can be quantized to 8-bit (**W8**), 4-bit (**W4**), 2-bit (**W2**), ternary weights (**W1.58**), or even binary (**W1**) weights. Using fewer bits for weights reduces the memory footprint to store the model data (ROM or Flash in a MCU).

**Quantization granularity of activations** - The number of bits used to encode the activations, the data as it progresses through the neural network. Inference is usually performed with floating point activations. But to reduce memory footprint and increase speed, activations can be quantized to 16-bit integers (**A16**), 8-bit (**A8**) or 1-bit (**A1**) in the most extreme case. Reducing the size of activations helps to preserve RAM during inference.

The most extreme case is to quantize both weights and activations to one bit (**W1A1**), as in the XNOR Net[^3]. However, this approach requires increasing the network size dramatically to counteract the loss of information from the activations. Also, handling of single bit information is more cumbersome on a standard 32-bit MCU than handling integers. 

Therefore, I will explore scenarios with different weight quantizations, while I keep the activations at 8 bit or more. It seems obvious that QAT is the preferred training approach when targeting inference on a very small microcontroller, but I will also explore post-quantization for comparison.

# Implementation of training code

I used the code snippets given in [^4] as a starting point. The main forward pass function remained unchanged. It implements a Straight-Through-Estimator (STE)[^5] to allow backpropagation through the non-differentiable quantization functions. Essentially, the weights are quantized during forward and backward pass, but the gradients are calculated with respect to unquantized weights. 

```python
def forward(self, x):
    x_norm = self.Normalize(x)

    w = self.weight # a weight tensor with shape [d, k]

    if self.QuantType == 'None':
        y = F.linear(x_norm, w)
    else:
        # A trick for implementing Straight-Through-Estimator (STE) using detach()
        x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
    return y
```

I implemented custom normalization and quantization functions to explore different options. I found no benefit for using batch normalization instead of RMS normalization[^6], hence I used the latter one as in the original BitNet implementation. This choice also simplifies the on-device inference. 

The implementation of one bit quantization is straight forward, the scale derived from the mean value of the weights in the entire layer.

```python
    scale = w.abs().mean().clamp_(min=1e-5)
    e = w.mean()
    u = (w - e).sign() * scale
```

For 2 and more bit quantization I chose to use symmetric encoding without zero, e.g. **[11,10,00,01] -> [+1.5 +0.5 -0.5 -1.5]**. Asymmetric encoding including zero did not show any benefits. The scale factor of 1.5 was chosen empirically. Experiments with per-channel scaling did not show any improvement over per-layer scaling in QAT, surely something to revisit later. The quantization function is shown below. 

```python
    mag = w.abs().mean().clamp_(min=1e-5)
    scale = 1.0 / mag # 2 worst, 1 better, 1.5 almost as bad as 2
    u = ((w * scale - 0.5).round().clamp_(-2, 1) + 0.5) / scale
```
To keep things simple, especially on side of the inference engine, I decided to use only fully connected layers and no CNN layers. To reduce memory footprint, the samples of the MNIST dataset are rescaled from 28x28 to 16x16 in size. This reduced resolution and lack of CNN layers will hamper achievable accuracy. But acceptable performance is still achievable, as shown later. The model structure is shown in the figure below.

<div align="center">
    <img src="./Model.drawio.png" width="60%">
</div>

The sizes of the hidden layers are parametrizable. 

# Model Optimization

The MNIST dataset with standard train/test split was used. Unless noted otherwise, the model was trained with a batch size of 128 and initial learning rate of 0.01 for 30 Epochs. The learning rate was reduced by a factor of 0.1 after each 10 epochs. Adam optimizer was used and cross entropy loss function.

## Quantization Aware Training vs Post-Quantization

To investigate the efficacy of QAT, I trained the model with different bit-widths for weights and quantized it to the same or smaller bit-width before testing. I kept the network size at *w1=w2=w3=64* for all runs. 

<div align="center">
    <img src="./prepostquant.png" width="60%">
</div>

Around 98% accuracy is achieved with 8-bit quantized weights, whether trained with QAT or post-quantization. When a model is quantized to 4-bits or less after training, a significant drop in accuracy is observed. Quantization aware training can distribute the quantization error to other weights and achieves good accuracy even for one bit weights. 

Advanced post-quantization schemes as used in LLMs could improve post-quantization accuracy, but QAT is clearly the preferred approach for this use case.

### Model Capacity vs Quantization scaling

To deploy the model on a microcontroller, the model size should be kept as small as possible. The required memory footprint is proportional to the number of weights and the quantization level of the weights. To understand all tradeoffs I trained a large number of models with different widths and quantization levels, all using QAT. Typically, the width of all layers was kept the same. 

<div align="center">
    <img src="./train_loss_vs_now.png" width="80%">
</div>

The plot above shows training loss vs. total number of weights. We can see that there is a polynomial relationship between the number of weights and the training loss. Reducing the number of bits per weight increases the loss proportionally. Interestingly, there are diminishing returns when increasing the number of bits beyond 4 and loss is not reduced further. It appears that beyond 4 bit, no more information per weight can be stored in the model. 

<div align="center">
    <img src="./train_loss_vs_totalbits.png" width="80%">
</div>

To understand which quantization level is the most memory efficient, I plotted the training loss vs. the total number of bits in the model. The number of bits is the product of the number of weights and the number of bits per weight. As can be seen in the plot above, the loss is proportional to the total number of bits, almost regardless of the quantization level between 1 and 4 bits. 

This trend seems to indicate that it is possible to trade bits per weight for an increased number of weights. Assuming that the training loss is indicative of the density of information stored in the model, this indicates that each bit carries the same amount of information, regardless of whether it is spent on increasing quantization levels or introducing more weights. 

```<rant>``` I am really intrigued by this result. Although this scaling relationship intuitively makes sense, I was not able to find much information about it. It raises the question of how to maximize information storage in the neural network. A cursory analysis of the weights according to Shannon information entropy suggests that it is near capacity for QAT, indicating that all weight encodings are used equally often. However, I also found that post-quantization of 4 bits can achieve the same loss with a weight encoding that follows a more normal distribution and therefore has lower entropy. If the entropy of the weights is not maximized, it means there could be on-the-fly compression to improve memory footprint. There are plenty of interesting things to come back to for later investigation.```</rant>```

Practically, this scaling relationship means that the number of bits per weight between 1-4 is a free variable that can be chosen depending on other requirements, such as compute requirements for inference.

### Test Accuracy and Loss

The scaling relationship above allows predicting train loss from model size. The plots below show the relationship between train and test loss and accuracy vs. model size.

<div align="center">
    <img src="./train_vs_test_loss.png" width="70%">
    <img src="./train_vs_test_accuracy.png" width="70%">
</div>

The tests above reveal a clear monotonic relationship between the model size and both test loss and accuracy, given the training setup used. However, there's a point of saturation where the test loss does not decrease any further. This could be indicative of overfitting, or it may suggest that the training data lacks sufficient variability to accurately represent the test data.

Considering the memory constraints of the target platform for inference, I've assumed that a maximum of 12kb of memory is available for the model. This is indicated by the red line in the plots above.

To improve model performance further, I fixed the model size to 12kb and explored various training parameters.

## Optimizing training parameters

The table below shows a set of network parameters that result in a model size close to 12 kbyte for various quantization levels. The number of weights in each layer was chosen to align parameter storage with int32. I tried to maintain the same width across all layers. 

<div align="center">

| Quantization [bit]  | 1 bit      | Ternary       | 2 bit       | 4 bit      | 8 bit       |
|---------------------|------------|---------------|-------------|------------|-------------|
| input               | 256        | 256           | 256         | 256        | 256         |
| w1                  | 176        | 128           | 112         | 64         | 40          |
| w2                  | 160        | 128           | 96          | 64         | 32          |
| w3                  | 160        | 112           | 96          | 64         | 32          |
| output                 | 10         | 10            | 10          | 10         | 10          |
| Number of weights   | 100416     | 64608         | 49600       | 25216      | 12864       |
| total bits [kbit]   | 100416     | 103372.8      | 99200       | 100864     | 102912      |

</div>

### Learning rate and number of epochs

I trained the models with different learning schedules and for a varying number of epochs. Switching from a linear to a cosine schedule resulted in a modest improvement, hence I kept this throughout the rest of the experiments.

<div style="text-align: center;">
    <img src="./12kopt_train.png" width="90%">
</div>

The training loss and accuracy is shown above. As expected from the experiments above, all models perform similarly. The 8bit quantized model has the higher loss, again. We can see that longer training reduces the loss and increases the accuracy.

<div align="center">
    <img src="./12kopt_test.png" width="90%">
</div>

The test loss and accuracy, in contrast, does not show a significant improvement with longer training. The test loss increases with longer training time, suggesting overfit of the training data.

<p align="center">
    <img src="./12kLoss_test.svg" width="60%">
</p>

The test loss plot above for the 120 epoch runs clearly shows that the higher the number of bits per weight, the greater the increase in test loss. This dependence is somewhat surprising, as one might assume from the previous results that all models have the same capacity and therefore should exhibit similar overfitting behavior. It has been previously suggested that low bit quantization can have a regularizing effect on the network [^7]. This could explain the observed behavior.

However, despite the regularizing effect, the test accuracy does not exceed 98.4%, suggesting that the model is unable to generalize to all of the test data.

### Data Augmentation

To improve the generalization of the model, and counter the overfitting, I applied data augmentation to the training data. Randomized affine transformations were used to add a second set of training images to the unaltered MNIST dataset. In each epoch, the standard set of 60000 training images plus 60000 images with randomized transformations were used for training.

```python
    augmented_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),  
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),   
        transforms.Resize((16, 16)),  # Resize images to 16x16
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
```	

<div align="center">
    <img src="./12kopt_augmented.png" width="90%">
</div>

The loss and accuracy for different learning run epochs are displayed above. The training loss is higher than that observed without data augmentation, but the test accuracy also increases by more than 0.5%, approaches 98.9%. The test loss decreases with a higher number of epochs, suggesting that the model is not yet overfitting the data. This is further confirmed by the test loss plot below. Inserting dropout layers was also able to reduce overfitting, but I found that data augmentation was more effective in improving the test accuracy.

<p align="center">
    <img src="./Aug_Loss_test.svg" width="60%">
</p>

Interestingly, the test accuracy trend is now reversed, with higher bit quantization showing a slight advantage, despite identical total model sizes. The reason for this is not entirely clear.

<p align="center">
    <img src="./explorationaugmented.png" width="80%">
</p>

The plot above shows further model structure exploration with fine-tuned data-augmentation and 60 epochs. Notably, the monotonic relationship between test accuracy and model size is even maintained when the depth is reduced from 4 layers to 3 layers. The labels show the widths of the hidden activation layers and the quantization level.

There are two exceptions, labels highlighted in bold:

1. The 8-bit quantization deviates from the trend, even with extended training. Judging from the gap to the trend, it appears that between 5 and 6 bits of the 8-bit parameters are effectively used.

2. The structure with a tapered width (64/48/32/4b) seems to introduce a bottleneck that reduces accuracy.


I was able to achieve >99% accuracy with 4-bit quantization after slightly tweaking the data augmentation parameters. The best trade-off appears to be the 64/64/64/4b structure. Further improvements might require a different model architecture, such as a CNN. However, to keep things simple, I will stop here. 99.0% accuracy already surpasses most (if not all) other MNIST inference implementations I have seen on low-end MCUs such as AVR.

## Summary of Learnings from Training
 
* Quantization Aware Training (QAT) enables high-accuracy models with low-bit quantization.
* Between 1 and 4-bit quantization, it was possible to trade the number of bits per weight for more weights without a significant impact on accuracy and loss. 
* 8-bit quantization was less efficient in terms of memory efficiency.
* Lower bit quantization can have a regularizing effect on the network, leading to less overfitting.
* When overfitting was addressed with data augmentation, higher bit quantization showed increasingly improved accuracy.
* No specific benefit of ternary encoding over symmetric 2 or 4-bit encoding was observed.
* Considering all the trade-offs above, it appears that 4-bit quantization is the most preferable option for the problem at hand, as it offered the best performance and required the least number of weights, reducing computational effort.
* 99.0% accuracy was achieved with 4-bit quantization and data augmentation. 

## Addendum: Visualization of First Layer Weights

I visualized the weights of the first layer of the model trained with data augmentation. There are 16x16 input channels and 64 output channels. We can see that each channel detects certain structured features of the full input image.

<p align="center">
    <img src="./first_layer_weights.png" width="60%">
</p>

In contrast, the visualization below represents the first layer weights of the model trained without data augmentation. The weights seem less structured and appear more random. Instead of learning general features, the network seems to tend to fit directly to the images, as suggested by the discernible shapes of numbers.

<p align="center">
    <img src="./first_layer_weights_noaugment.png" width="60%">
</p>

## Addendum: Potential of Additional CNN layers

Convolutional Neural Networks (CNNs) are typically preferred for image processing over fully connected networks due to their better performance. However, their implementation is more complex, which is why they were not included in this exploratory project.

The weights of a CNN are exposed to many different features within the same image, enabling them to learn generalized features much more effectively. This makes them more robust against overfitting and reduces the need for data augmentation. Additionally, they trade higher computational effort for a smaller memory footprint, as the weights are shared across the image.

To explore the potential of adding CNN layers, I added two 3x3 Conv2D layers to the model. The Conv2d layers were trained in float, while the fc layers were trained with 4-bit QAT. The model was trained with the same parameters as before (w1=w2=w3=64, augmentation, 60epochs).

```python 
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
...
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    x = F.relu(self.conv2(x))        
    x = F.max_pool2d(x, kernel_size=2, stride=2)
```

This modification increased the test accuracy to over 99%. Memory-efficient depthwise convolution with similar parameters yielded comparable results. Another topic to follow up opon later...

# Architecture of the Inference Engine

Since the training is performed on the model in float format (the weights are only quantized during forward and backward pass), the model needs to be converted to a quantized format and then exported to a format that can be included into the inference code on the CH32V003.

Let's first start with the target architecture. To reduce computational effort on the microcontroller as much as possible, I modified the network architecture implementation.

Key observations:

- No normalization is needed for the first layer as the input data (test sample) is already normalized.
- The fully connected layer accepts 8 bit signed integers, but as a result of the summation of partial products, the output is 32 bit.
- The normalization operation rescales the output back to an 8-bit value, but can only do this once all activations from the previous layer are available.
- Classification is performed by selecting the maximum value from the output layer and returning the index, which needs to be done before the ReLU operation.

These requirements can be met by the network architecture below. Each layer consists of a fully connected layer, followed by a fused block where normalization and ReLU are combined. Classification (maxpos) is performed before the ReLU operation in the same block, but the result is only used for the output layer.

<div align="center">
    <img src="Model_mcu.drawio.png" width="80%">
</div>
 
One crucial observation for further optimization of the inference engine:

<p align="center">
<strong>The model is tolerant to scaling errors as it doesn't employ any operations, other than ReLU, that alter the mean or linearity.</strong>
</p>

This is true because RMS norm is used as opposed to batch or layer norm, which prevents altering the mean of the activations. Additionally, no bias was used. While ReLU sets negative values to zero, it maintains the linearity of positive values.

This allows for a simpler normalization scheme to replace the RMS norm. I discovered that a simple shift operation, **ShiftNorm**, could be used instead. This operation shifts all output values in the layer until the maximum is less than 127 (int8_t). While this introduces some quantization noise, they I found it inconsequential in testing.

Ultimately, this means that no multiplication operation is required outside of the fully connected layer!

## Implementation in Ansi-C

### fc-layer

The code for a convolution with binary weights is straightforward:

```c
    int32_t sum = 0;
    for (uint32_t k = 0; k < n_input; k+=32) {
        uint32_t weightChunk = *weightidx++;

        for (uint32_t j = 0; j < 32; j++) {
            int32_t in=*activations_idx++;
            sum += (weightChunk & 0x80000000) ? in : -in; 
            weightChunk <<= 1;
        }
    }
    output[i] = sum;
```

This is how the inner loop looks in RV32EC assembly (compiled with -O3)

```asm
<processfclayer+0x28>
     1d8:	0785                addi	a5,a5,1
     1da:	fff78303          	lb	    t1,-1(a5) # ffff <_data_lma+0xc6db>
     1de:	02074563          	bltz	a4,208    <processfclayer+0x58>
     1e2:	0706                slli	a4,a4,0x1
     1e4:	40660633          	sub	    a2,a2,t1
     1e8:	fe5798e3          	bne	    a5,t0,1d8 <processfclayer+0x28>
    ...
<processfclayer+0x58>
     208:	0706                slli	a4,a4,0x1
     20a:	961a                add	    a2,a2,t1
     20c:	fc5796e3          	bne	    a5,t0,1d8 <processfclayer+0x28>    
```

The full loop is 6 instructions while the actual computation is just 3 instructions (lb, bltz,neg/add). The compiler did quite a good job to split the conditional into two code paths to avoid an addition "neg" instruction. 

It would be possible to unroll the loop to remove loop overhead. In that case 4 instructions are required per weight, since the trick with two codes paths would not work easily anymore.

Convolution with 4 bit weight is shown below. The multiplication is implemented by individual bit test and shift, as the MCU does not support a native multiplication instruction. The encoding as one-complement number without zero helps with code efficiency. 

```c
    int32_t sum = 0;
    for (uint32_t k = 0; k < n_input; k+=8) {
        uint32_t weightChunk = *weightidx++;

        for (uint32_t j = 0; j < 8; j++) {
            int32_t in=*activations_idx++;
            int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; 
            sum += tmpsum;                                  // sign*in*1
            if (weightChunk & 0x40000000) sum += tmpsum<<3; // sign*in*8
            if (weightChunk & 0x20000000) sum += tmpsum<<2; // sign*in*4
            if (weightChunk & 0x10000000) sum += tmpsum<<1; // sign*in*2
            weightChunk <<= 4;
        }
    }
    output[i] = sum;
```
Again, the compiled code of the inner loop below. The compiler decided to unroll the loop (8x), which removed the loop overhead.

```asm
     1d2:	00060383          	lb	    t2,0(a2)
     1d6:	00075463          	bgez	a4,1de <processfclayer+0x2e>
     1da:	407003b3          	neg	    t2,t2
positive:

    	01cf5a93          	srli	s5,t5,0x1c
    	007afa93          	andi	s5,s5,7
    	015898b3          	sll	    a7,a7,s5
    	9846               	add	    a6,a6,a7

    	0e85               	addi	t4,t4,1
    	0f12               	slli	t5,t5,0x4

	fdfe9fe3          	bne	    t4,t6,20000158 <loop>
```

In total 17 instructions are required per weight, with no additional loop overhead. 

Considering the observations during model optimization, binary weights require approximately four times as many weights as 4-bit quantization to achieve the same performance. The execution time for binary inference is `4 cycles * 4 * number of weights`, while for 4-bit quantization, it's `17 cycles * number of weights`.

Consequently, the pure computation time is comparable for both quantization levels, offering no inference time advantage for binary weights for the given problem setting. In fact, due to the additional overhead from the increased number of activations required with binary weights, the total execution time is likely higher for binary weights.

The implementation for 2 bit quantization is not shown here, but it is similar to the 4 bit quantization. I did not implement Ternary weights due to complexity of encoding the weights in a compact way.

It should be noted, that the execution time can be improved by skipping zero activations. Typically, more than half of the activations are zero.

### ShiftNorm / ReLU block

The fused MaxPos / ShiftNorm / ReLU block is straightforward to implement. 

```c
    // Find the maximum value in the input array
    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_pos = i;
        }
    }

    // Find shift value to normalize the maximum value to <127 
    tmp=max_val>>7; 
    scale=0;

    while (tmp>0) {
        scale++;
        tmp>>=1;
    }

    // Apply ReLU activation and normalize to 8-bit
    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] < 0) {
            output[i] = 0;
        } else {
            output[i]=input[i] >> scale;  
        }    
    }
    return max_pos;
```
And that's all - **not a single multiplication operation was required**. 

# Putting it all together 

To keep things flexible, I split up the data pipeline into several python scripts. **training.py** is used to train the model and store it as *.pth* file. The model weights are still in float format at that time, since they are quantized on-the-fly during training. **exportquant.py** converts the model into a quantized format, a custom python class, that is only used as an intermediate representation for export and testing. The quantized model data is then merged into 32 bit integers and exported to a C header file. 

To test inference of the actual model as a C-implementation, the inference code along with the model data is compiled into a DLL. **test-inference.py** calls the DLL and compares the results with the original python model test case by test case. This allows accurate comparison to the entire MNIST test data set of 10000 images. 

The flow is shown in the figure below.

<div align="center">
    <img src="trainingpipeline.png" width="60%">
</div>

## Model Exporting

Output of the exporting tool is shown below. Some statistics on parameter usage are printed. We can see that the model is using all available codes, but they are not evenly distributed. This means that the model could be compressed further and that the entropy is not maximized.  

```
Inference using the original model...
Accuracy/Test of trained model: 98.98 %
Quantizing model...

Layer: 1, Max: 7.5, Min: -7.5, Mean: -0.00360107421875, Std: 2.544043042288096
Values: [-7.5 -6.5 -5.5 -4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5  5.5 6.5  7.5]
Percent: [ 2.47192383  0.73242188  0.8972168   1.59301758  3.08227539  6.68334961 12.83569336 20.71533203 20.86181641 13.9831543   7.36083984  3.47900391 2.12402344  1.21459961  0.81176758  1.15356445]
Entropy: 3.25 bits. Code capacity used: 81.16177579491874 %

Layer: 2, Max: 7.5, Min: -7.5, Mean: -0.12158203125, Std: 2.5687543790088463
Values: [-7.5 -6.5 -5.5 -4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5  5.5 6.5  7.5]
Percent: [ 1.171875    0.95214844  1.53808594  3.56445312  5.54199219  8.71582031 12.25585938 16.38183594 16.50390625 13.50097656 10.05859375  5.56640625 2.41699219  1.09863281  0.46386719  0.26855469]
Entropy: 3.38 bits. Code capacity used: 84.61113322636102 %

Layer: 3, Max: 6.5, Min: -7.5, Mean: -0.23291015625, Std: 2.508764116126823
Values: [-7.5 -6.5 -5.5 -4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5  5.5 6.5]
Percent: [ 0.78125     0.9765625   1.92871094  4.05273438  5.71289062 10.08300781 12.93945312 14.23339844 16.33300781 14.23339844  9.91210938  5.59082031 2.34375     0.63476562  0.24414062]
Entropy: 3.35 bits. Code capacity used: 83.84599479239081 %

Layer: 4, Max: 4.5, Min: -7.5, Mean: -0.73125, Std: 2.269283683786582
Values: [-7.5 -5.5 -4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5]
Percent: [ 0.15625  0.78125  4.0625  12.96875 15.3125  15.625   14.6875  11.25 9.53125 10.46875  4.21875  0.9375 ]
Entropy: 3.15 bits. Code capacity used: 78.82800031261552 %
Total number of bits: 100864 (12.3125 kbytes)
inference of quantized model
Accuracy of quantized model: 99.00999999999999 %
```

The total size of the model is 12.3 kilobytes. The tool also performs inference on both the original PyTorch model and a quantized version that emulates the ShiftNorm operation. Interestingly, the accuracy of the quantized model is 99.01%, which is slightly better than the original model. The model data is written to a header file, which is subsequently included in the C code.

```c
const uint32_t L2_weights[] = {0x1231aa2, 0x29c09a90, 0x20a16b50, 0x8c938109, 0x320a2301, 0x2810008, 0x89114a9a, 0x9fb1c101, 0x899c90, 0x2889329, 0xab0b9bcc, 0x9a419319, 0x8209091a, 0x2b8da0b9, 0x282144a0, 0x3fb8881, ...
```

## Verification of the Ansi-C Inference Engine vs. Python

The exported data and the inference engine are compiled to DLL, which is then called from the python test script and compares the predictions image- by image.

The output is shown below. Curiously, the C inference engine is yet again slightly better than the Python implementation. There are still three (out of 10000) test images where both engine disagree. I believe this is due to different rounding behavior of the two engines. I was already able to reduce this from a larger number by adding additional rounding to the ShiftNorm operation. 

```
Loading model...
Inference using the original model...
Accuracy/Test of trained model: 98.98 %
Quantizing model...
Total number of bits: 100864 (12.3125 kbytes)
Verifying inference of quantized model in Python and C
Mismatch between inference engines. Prediction C: 6 Prediction Python: 4 True: 4
Mismatch between inference engines. Prediction C: 5 Prediction Python: 0 True: 5
Mismatch between inference engines. Prediction C: 3 Prediction Python: 5 True: 3
size of test data: 10000
Mispredictions C: 98 Py: 99
Overall accuracy C: 99.02 %
Overall accuracy Python: 99.00999999999999 %
Mismatches between engines: 3 (0.03%)
```
## Implementation on the CH32V003

The implementation on the CH32V003 is straightforward and can be found [here](https://github.com/cpldcpu/BitNetMCU/tree/main/mcu). The model data is included in the C code, and the inference engine is called from the main loop. I used the excellent [CH32V003fun](https://github.com/cnlohr/ch32v003fun) environment to minimize overhead code as much as possible. This allowed me to include up to 12KB of model data into the 16KB of flash memory.  The execution timing was optimized by moving the fc-layer code to the SRAM, which avoids flash wait states. Further optimizations on assembler level will certainly improve the performance further, but the generated code was already quite good.

<div align="center">
    <img src="themcu.jpg" width="50%">
</div>

Four test cases are evaluated, and the execution timing is measured using the internal SysTick profiling timer. The results are printed to the debug-console using printf.

Example output for inference with a 25126 4-bit parameter model is shown below.

<div align="center">
    <img src="../mcu/console.png" width="80%">
</div>

The execution time is approximately 650,000 cycles, which corresponds to 13.66ms at a 48MHz main clock. This is equivalent to 3.69 million operations per second ("MOPS"). The model achieves a test accuracy of 99.02%, which is quite impressive for such a small microcontroller and surpasses all other MCU-based MNIST implementations I have encountered.

I also tested a smaller model with 4512 2-bit parameters. Despite its size, it still achieves a 94.22% test accuracy. Due to its lower computational requirements, it executes in only 1.88ms.

# Summary and Conclusions
This marks the end of my journey to implement an MNIST inference engine with an impressive 99.02% test accuracy on a very limited $0.15 RISC-V microcontroller, which lacks a multiplication instruction and has only 16KB of flash memory and 2KB of RAM.

This achievement was made possible by employing Quantization Aware Training (QAT) in connection with low-bit quantization and precise model capacity tuning. An interesting observation, which I had not seen so clearly elsewhere, is that at the capacity limit, it's feasible to trade the number of bits per weight for the number of weights. The inference accuracy could be predicted by the total number of bits used for the weights within certain limits. There was a slight improvement in performance when using 4-bit quantization over lower bit quantization. 8-bit quantization offered diminishing returns and only stored about 5 bits of equivalent information per weight.

By simplifying the model architecture and using a full-custom implementation, I bypassed the usual complexities and memory overhead associated with Edge-ML inference engines.

While this project focused on MNIST inference as a test case, I plan to apply this approach to other applications in the future.
# Updates
## May 20, 2024: Additional quantization schemes

This section outlines additional quantization schemes that improve inference speed to microcontrollers without and with multiplier. WCH has recently announced new members of the CH32V003 family that come with a slightly extended instruction set architecture, RV32EmC or officialle RV32EC-Zmmul, which also support multiplication. It is likely that the CH32V003 will remain the only multiplierless RISC-V MCU in the industry, hence supporting multiplications is a good idea.

### FP1.3.0 Quantization

FP1.3.0 or FP130 is a quantization scheme based on 4-bit floating point numbers with 1-bit sign, 3-bit exponent and 0-bit mantissa. Weights are encoded as follows: $w = \text{sign} \times 2^{\text{exponent}}$. This will provide us with weights as exponents of two without zero: ```-128, -64 ... -2, -1, 1, 2, ... 64, 128```

The implementation of the inference code in C is extremely effective as only shift operations are required:

```c
    for (uint32_t k = 0; k < n_input; k+=8) {
        uint32_t weightChunk = *weightidx++;
        for (uint32_t j = 0; j < 8; j++) {
            int32_t in=*activations_idx++;
            int32_t tmpsum;
            
            tmpsum = (weightChunk & 0x80000000) ? -in : in;  // sign 
            sum += tmpsum << ((weightChunk >> 28) & 7);      // sign*in*2^log                       
            weightChunk <<= 4;
        }
```

Accordingly, the code compiled to only a few instructions per weight, even on RV32EC.

```asm	
loop:
	01db08b3          	add	    a7,s6,t4

	00088883          	lb	    a7,0(a7)
	000f5463          	bgez	t5,20000168 <positive>
	411008b3          	neg	    a7,a7
positive:

	01cf5a93          	srli	s5,t5,0x1c
	007afa93          	andi	s5,s5,7
	015898b3          	sll	    a7,a7,s5
	9846               	add	    a6,a6,a7

		0e85               	addi	t4,t4,1
		0f12               	slli	t5,t5,0x4

	fdfe9fe3          	bne	    t4,t6,20000158 <loop>
```

Amazingly, Quantization Aware Training is able to adjust the weights in a way where this encoding can be used efficiently. A test accuracy of 98.66% was achieved with the same model size and training settings, which is only slightly lower than for ```4bitsym``` encoding. The inference time reduces to 10.17ms from 13.66ms due to the simpler shift operation.

This is quite remarkable as using shifts instead of multiplications also would reduce complexity (circuit size) on dedicated inference hardware significantly. There seems to be some research on similar quantization schemes[^8], but no broad adoption yet.

The first layer weights are shown below. Due to the increased contrast enforced by the exponential encoding, we can see stronger differences between patterns.

<div align="center">
    <img src="first_layer_weights_fp130.png" width="60%">
</div>

The entropy is comparable to other 4 bit encodings, suggesting similar effective use of the coding space. We can, however, see that the lower layers do not use all of the available codes, which could be optimized further but different normalization schemes.

<div align="center">
    <img src="fp130_export.png" width="80%">
</div>


### 4-bit ones complement quantization

The current implementation of 4 bit quantization ```4bitsym``` uses a symmetric encoding without zero. This is easy to implement on multiplierless MCUs, but becomes unnecessarily complex when a multiplier is available. Therefore, I introduced ```4bit``` encoding, which encodes a 4 bit signed value is a one-complement number including zero: ```-8, -7 ... -2, -1, 0, 1, 2, ... 6, 7```.

This allows for a more efficient implementation of the inference code, given that the multiplication instruction is available:

```c
    for (uint32_t k = 0; k < n_input; k+=8) {
        int32_t weightChunk = *weightidx++;
        for (uint32_t j = 0; j < 8; j++) {
            int32_t in=*activations_idx++;
                            // extend sign, remove lower bits
            int32_t weight = (weightChunk) >> (32-4); 
            sum += in*weight;                                  
            weightChunk <<= 4;
        }
```

Compiles to the following, much shorter, assembly code: 

```
loop:
    	01ca8f33          	add	    t5,s5,t3
    	000f0f03          	lb	    t5,0(t5)

    	41cedb13          	srai	s6,t4,0x1c
    	036f0f33          	mul	    t5,t5,s6
    	987a                add	    a6,a6,t5

    	0e05                addi	t3,t3,1
    	0e92                slli	t4,t4,0x4

    	fffe15e3          	bne	    t3,t6,2000011e <loop>
```
## May 20, 2024: Quantization scaling

I introduced a new hyperparameter that was previously hardcoded: ```quantscale```. This parameters influences the scaling of the weights. It will determine the value of the standard-deviation of the weights per tensor relative to the maximum value of the quantization scheme. Previously, the parameter was set to a default of 0.25, which corresponds to a standard deviation of approximately 2 for the ```4bitsym``` encoding.

The plot below shows how the parameter influences the distribution of the first layer weights for the ```4bitsym``` encoding. 

<div align="center">
    <img src="4bit_histograms.png" width="60%">
</div>

We can see that the weights follow roughly a normal distribution with some extreme outliers. Changing quantscale to a higher value with make the distribution wider and increase the fraction of outliers at the maxima. QAT makes sure that the errors introducing from clipping the outliers are distributed to other weights.

<div align="center">
    <img src="quantscale_scan.png" width="70%">
</div>

I performed a scan of the parameter for the ```4bitsym``` and ```4bit``` encoding. We see that too high (0.5) and too low (0.125) degrade the weight distribution, leading to an increase of loss and worse test and train accuracy. Within the range of 0.2 to 0.4, the performance seems to be relatively stable. However, there is still a strong random variation of accuracy, caused by different initializations of the weights. This is also owed to the marginal capacity of the model which was minimized as much as possible. 

<div align="center">
    <img src="quantscale_entropy.png" width="70%">
</div>

There is a rather interesting relationship when looking at standard deviation and [information entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) across the layers. As expected, ```quantscale``` biases the standard deviation in a roughly proportional way. However, we can also see that the entropy increases for higher values. For low settings, this is because most weights are around zero and are truncated. Increasing the scale parameter also increases entropy. However, the accuracy of the model does not benefit, which means that only noise is added and no useful information. 

Already for an entropy of around 3 bits, it is possible to roughly maximize accuracy. This suggests that the weights can be compressed further to less than 80% of their original size, for example with an additional [entropy coding step](https://en.wikipedia.org/wiki/Entropy_coding), without loss of accuracy. Its an interesting question, whether this can also be achieved by different weight encoding.

## July 19, 2024: OCTAV Optimum Clipping

In the previous update I introduced a new hyperparameter to control the quantization step size and indirectly the clipping. It is, however, also possible to determine an optimum based on the given weight distribution. Such a method, called OCTAV, is presented a recent paper by Nvidia (Sakr et al. [^9]). I found it via [this talk ](https://www.youtube.com/watch?v=gofI47kfD28) by Bill Dally, which is a recommended watch.

The method introduces a clipping parameter `s` that determines the maximum encoded weight-value. Values with a magnitude larger than s will be clipped to s, values with a smaller magnitude are quantized according to the quantization step size. 

The octav algorithm determines the clipping parameter by minimizing the mean squared error `J(s)`between the original and quantized weights: Weight below s contribute to the error with quantization noise `(s*2*2^-bpw)²/12 = s²*4^-bpw / 3`, weight above s with the clipping error `(weight-s)²`. 

<div align="center">
    <img src="octav_equation.png" width="50%">
</div>

The clipping parameter `s` that minimizes the error can be determined by iteration with the newton method[^9]. The implementation resides in `BitNetMCU.py` and is also shown below.

```python	
for _ in range(num_iterations):
    indicator_le = (torch.abs(tensor) <= s).float()
    indicator_gt = (torch.abs(tensor) > s).float()
    numerator = torch.sum(torch.abs(tensor) * indicator_gt)
    denominator = (4**-self.bpw / 3) * torch.sum(indicator_le) + torch.sum(indicator_gt)
    s = numerator / denominator
```

The octav method is called after every training epoch to adjust the clipping parameter for each layer. The evolution of s and entropy per layer vs training epoch is shown below.

<div align="center">
    <img src="octav.png" width="90%">
</div>

Compared to the empirical setting of `quantscale`, the octav method yielded a similar training loss. This means we can reduce the number of hyperparameters to tune.

It appears that octav minimizes the entropy of the weights, without affecting accuracy. This could be interpreted as reducing noise.

<div align="center">
    <img src="octav_weightdist.png" width="60%">
</div>

Looking at the distribution, it is curious that there are very few weights with clipped values at the extremes. 

## July 26, 2024: NormalFloat4 (NF4) Quantization

Normalfloat is a data type that was introduced in the QLoRa paper by T. Dettmers[^10]. The idea is to map 4-bit weights in a way where more values are available around zero, which is the most common value for weights. The data type is information-theoretically optimized for normally distributed weights.

<div align="center">
    <img src="NF4plot.png" width="50%">
</div>

The plot above shows the weight encoding. Typically, this datatype is used for post-quantization, but it also makes sense for QAT, since the weight distribution follows a normal distribution as well.

To implement this datatype, it is necessary to quantize values according to an encoding table. The Python implementation (proposed by 3.5-Sonnet) is shown below. Frankly, I am quite impressed by the implementation, which hardly increased training time.

```python
    ...
    elif self.QuantType == 'NF4':
        # NF4 levels (16 levels for 4 bits)
        levels = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 
                                0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.723, 1.0], device=w.device)
        u , _ = self.quantize_list(w * scale, levels)
    ...

    def quantize_list(self, x, levels):
    """
    Quantize the input tensor x to the nearest level in the levels list.
    """        
    # Compute the absolute difference between x and each level
    diff = torch.abs(x.unsqueeze(-1) - levels)
    # Find the index of the closest level for each element in x
    indices = torch.argmin(diff, dim=-1)

    return levels[indices], indices
```

Initial tests showed that `NF4` performed similarly to the linear 4-bit encoding `4bitsym`. To get a better assessment of the initial model capacity enabled by this datatype, I performed a scaling experiment where I varied the number of weights in the model by changing the width of the model in three steps (40, 48, 56). I intentionally kept the model size below capacity for MNIST (~64 width for the fc model) to avoid saturating the model capacity. I used short training runs (20 epochs) to save time.

<div align="center">
    <img src="NF4lossplots.png" width="80%">
</div>

Similar to the network scaling experiment above, we can now plot training loss vs. model capacity. The plot below shows the results for three different quantization schemes: `NF4`, `4bitsym`, and `FP130`.

<div align="center">
    <img src="NF4scaling.png" width="60%">
</div>

We see that the `NF4` encoding consistently leads to lower loss at the same network size than both `4bitsym` and `FP130`. `FP130` performs the worst, which is likely due to poor code use because of the exponential encoding.

To achieve the same loss, an `NF4` encoded model requires ~3% fewer parameters than `4bitsym`, while `FP130` requires ~10% more.

The benefit is rather small, most likely because quantization-aware training is generally very good at adapting to any quantization scheme.

I have not yet implemented C-based inference code for `NF4`; however, it would allow for efficient implementation with table lookups. For example, W4A4 would require a 256-entry table to multiply one weight with one activation, which is rather small. In that case, `NF4` encoding could also be used for activations.

## Aug 3rd, 2024: Stepped Learning Rate Schedule

The Bitnet paper used a learning rate schedule that[^4] with a stepwise reduction after half of the epochs. I implemented a similar behavior by adding a configuration option that halves the learning rate at a designated epoch. To deactivate this behavior, the parameter can be set to an invalid epoch or be commented out.

```
halve_lr_epoch: 30  # Epoch at which to halve the learning rate 
```
The plot below shows different learning rate schedules with and without step reduction at epoch 30 and a starting learning rate of 0.003 and 0.001. (Previous default was lr=0.001 and no halving). (NF4 encoding, 64-width model, 60 epochs total). Accuracy and loss curves for the same runs are shown on the right.
<div align="center">
    <img src="schedules.png" width="47%">  <img src="lossvsschedule.png" width="40%">
</div>

The halving leads to an immediate improvement in training loss; however the benefit compared to the runs without halving is lost at the end of the training run. Interestingly, the halving improves test loss. This suggests that the halving leads to better regularization. The benefits are rather small, though, and may have more effect in datasets where the model capacity is more limiting.

## Aug 31st, 2025: Porting to CH32V002 and newest version of CH32fun

I ported the example code from the CH32V003 to the CH32V002 and updated to the latest version of the CH32fun environment. It can now be compiled for any CH32V00X MCU (and also others from that series).

Compared to the CH32V003, the CH32V002 brings some minor changes, both good and bad.
- The RISC-V core was updated to a RV32EmC instruction set architecture, which includes a multiplication instruction.
- The Flash memory now requires 2 waitstates instead of 1 waitstate at 48MHz, reducing effective code execution speed from Flash.

### Execution Time Comparison between CH32V002 and CH3V003

I compared the execution times of both MCUs in all possible configurations, with code execution from SRAM or Flash and for code using the multiplication instructions instead of bitwise adding. This is for the 12kb model with 4bit symmetric weight encoding. (Avg cycles = mean of 3 samples; Time = cycles / 48e6.)

| MCU / Mode | Memory | Multiplier | Avg cycles | Relative |
|------------|--------|------------|-----------:|---------:|
| CH32V003   | SRAM   | no         |   653,965  | 1.00x |
| CH32V002   | SRAM   | no         |   677,362  | 1.04x |
| CH32V002   | SRAM   | yes        |   508,605  | 0.78x |
| CH32V003   | Flash  | no         |   903,209  | 1.38x |
| CH32V002   | Flash  | no         | 1,343,729  | 2.06x |
| CH32V002   | Flash  | yes        | 1,015,672  | 1.55x |

We see that even for execution from SRAM, the V002 is around 3.6% slower than the V003. This
is due to added waitstates in the parts of the code that were executed from Flash. However, when we use the multiplier, things change and we get a nice 22% speedup vs. V003. 

Code execution from flash degrades performance significantly for both V003 and V002. Due to the added waitstates, the V002 takes 48% longer, which is roughly proportional to the increase in memory access timing. The improvements from the fast multiplication can barely compensate for this. 

Note that the slowdown for execution from flash depends on the fraction of 16 bit instructions in the code, as 32 bit fetches may partially hide latency. Note that the V002 has 4kb SRAM vs. 3kb in the V003 which allows for more code to be executed from SRAM.

## Aug 15th, 2024 & Sep 15th, 2025: Convolutional Neural Networks (CNN)

Now that the CH32V002 support with a hardware multiplier is available, it is time to merge another feature that has been brewing in a branch for over a year: [Convolutional Neural Networks (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network). 

All previous models were based on fully connected layers, which basically perform a matrix-vector multiplication, where a weight is reserved for each combination of input to output. This architecture is very strong in learning exact patterns, however it does not possess any spatial awareness or inductive bias. It can learn a certain pattern at exactly one location, but will not recognize it if it is shifted by only one pixel.

CNNs, are based on the idea of learning filter kernels that are convoluted with the entire input image. When the image is slightly shifted or otherwise distorted, the filter will still recognize the pattern. It is translation invariant. 

Due to this, CNNs are vastly more powerful for image recognition tasks and were also the architecture that put machine learning on the map in the 90ies with [LeNet](https://en.wikipedia.org/wiki/LeNet)[^11] and ended the "AI-Winter" by leading to the breakthrough in deep-learning image recognition with DanNet[^12] and finally AlexNet[^13] in 2012.

But how do we implement them on a very limited low-end microcontroller? What is easy in PyTorch is not necessarily easy when doing it manually in C on a memory and compute limited device.

   
1) CNN tend to increase the number of channels (feature maps) as progressing through the layers. Even for a 16x16 image, we need 512 or 1024 bytes per channel if we store them in parallel. This quickly exceeds the available RAM of only 3 or 4kB.

2) Convolutional layers are parameter efficient, but computationally very expensive. Every parameter in the kernel is multiplied with every pixel in the input image. This leads to a large number of operations, which can be easily parallelized on a GPU, but are very time consuming on a single-core MCU.

3) The convolution operation requires a lot of special case handling like padding, stride, and dilation. This makes the implementation more complex and increases code size.

### Model architecture

Following a lengthy "human-assisted architecture search", I settled with the following architecture:

<div align="center">
    <img src="Model_cnn_overview.png" width="70%"> 
</div>

The first layer of the former fully connected model is replaced with convolutional layers that perform the function of extracting features from the input image. 

### Deep-Depthwise Convolution and in-place processing

The implementation of the CNN layers is a bit unusual. Typically, a CNN is evaluated layer-by-layer. However, when the number of channels is expanded, this results in quickly increasing memory requirements. For example, to store the activations of a 16x16 images with 32 bit precision and 64 channels, we would need 16*16*4*64B = 65536B = 64kB of RAM, which exceeds the capabilities of the target device by far. In addition, performing convolutions across all channels requires a lot of computational power - 16x16x3x3x64x64 = 9,437,184 multiplications for just one layer.

To avoid this, a technique called depthwise separable convolutions had been introduced for architectures targeting mobile devices, prominently used in MobileNet[^14] and Xcpetion[^15]. Here, the convolution is split into two separate operations: a depthwise convolution that operates on each channel separately and a pointwise operation that combines the outputs. This reduces the number of parameters and computations significantly. For the example above, the number of multiplications reduces to 16x16x3x3x64 + 16x16x64x64 = 1,196,032, which is signifantly less.

We take this concept one step further and use deep depthwise separated convolutions, where we perform multiple depthwise convolutions and maxpooling operations in sequence.

<div align="center">
    <img src="Model_cnn_channel.png" width="40%"> 
</div>

The image above shows the architecture of a single CNN channel. The input image is processed by three convolution layers with ReLU activation and two maxpooling layers. The output is a 2x2=1x4 feature map. A total of 64 channels are processed in parallel and the resultsing 64x2x4 = 256 values are then fed into the fully connected layers.

This approach has the advantage that each channel can be processed separately and only the 4 final activitation needs to be stored. In addition, we can use in-place processing[^16] where the output of each layers replaces the input data in the same memory array. This allows us to reduce the memory footprint of the convolution operations to that of a single channel: 16x16x4B = 1kbyte.

### Simplifying convolutions and implementation

Addressing the third point: To reduce complexity, I opted to go for the simplest possible conv2d operations: A 3x3 kernel with stride 1 and no padding. This allows for a greatly simplified implementation as no special case handling is required.

A potential issue is that patterns at the edge of the image cannot be detected well. However, as shown in the figure below, the edge pixels are almost never used in the MNIST dataset. (Left: Probability of each pixel being part of a digit in both the MNIST train and test set. Right: Black indicates pixels that are never used.)

<div align="center">
    <img src="combined_pixel_analysis.png" width="70%"> 
</div>

The code of the simplified convolution operation is shown below. It is unrolled for a 3x3 kernel and is fused with a ReLU activation function. 

```c
int32_t* processconv33ReLU(int32_t *activations, const int8_t *weights, uint32_t xy_input, uint32_t  n_shift , int32_t *output) {

    for (uint32_t i = 0; i < xy_input - 2; i++) {
        int32_t *row = activations + i * xy_input;
        for (uint32_t j = 0; j < xy_input - 2; j++) {
            int32_t sum = 0;
            int32_t *in = row ++;

            // Unrolled convolution loop for 3x3 kernel
            sum += weights[0] * in[0] + weights[1] * in[1] + weights[2] * in[2];
            in += xy_input;
            sum += weights[3] * in[0] + weights[4] * in[1] + weights[5] * in[2];
            in += xy_input;
            sum += weights[6] * in[0] + weights[7] * in[1] + weights[8] * in[2];

            // Apply shift and ReLU
            if (sum < 0) {
                sum = 0;  // ReLU
            } else {
                sum = sum >> n_shift;
            }
            *output++ = (int32_t)sum;
        }
    }
    return output;
}
```

The other primitive that is required for CNNs is max-pooling, which reduces the spatial dimensions of the image by taking the maximum value in a 2x2 patch. 

```c
int32_t *processmaxpool22(int32_t *activations, uint32_t xy_input, int32_t *output) {
    uint32_t xy_output = xy_input / 2;

    // Iterate over the output array dimensions
    for (uint32_t i = 0; i < xy_output; i++) {
        int32_t *row = activations + (2 * i) * xy_input;
        for (uint32_t j = 0; j < xy_output; j++) {            

            // Find the maximum value in the corresponding 2x2 patch in the input activations
            int32_t max_val;
            max_val = row[0];
            max_val = max_val > row[xy_input] ? max_val : row[xy_input];
            row++;
            max_val = max_val > row[0] ? max_val : row[0];
            max_val = max_val > row[xy_input] ? max_val : row[xy_input];
            row++;

            // Store the maximum value in the output array
            *output++ = max_val;
        }
    }
    return output;
}
```

<div align="center">
    <img src="model_cnn_mcu.png" width="70%"> 
</div>

- Simplified conv2d implementation
- Variable quantization


- Deep depthwise convolutions
- In-place processing of channels





binary
Total number of bits: 90112 (11.0 kbytes)
Verifying inference of quantized model in Python and C
  247 Mismatch between inference engines found. Prediction C: 6 Prediction Python: 2 True: 4
 3023 Mismatch between inference engines found. Prediction C: 8 Prediction Python: 5 True: 8
size of test data: 10000
Mispredictions C: 46 Py: 47
Overall accuracy C: 99.53999999999999 %
Overall accuracy Python: 99.53 %
Mismatches between engines: 2 (0.02%)

![alt text](image-3.png)

64 wide
2bit mlp

Epoch [1/60], LTrain:0.288631 ATrain: 91.54% LTest:0.069724 ATest: 97.97% Time[s]: 19.72 w_clip/entropy[bits]: 0.614/7.14 0.554/7.24 0.646/7.10 0.137/1.81 0.180/3.71 0.372/3.78 
Epoch [2/60], LTrain:0.115796 ATrain: 96.40% LTest:0.041229 ATest: 98.74% Time[s]: 19.94 w_clip/entropy[bits]: 0.615/7.21 0.617/7.14 0.730/7.02 0.184/1.77 0.220/3.57 0.426/3.77 
Epoch [3/60], LTrain:0.085445 ATrain: 97.35% LTest:0.037770 ATest: 98.79% Time[s]: 19.99 w_clip/entropy[bits]: 0.707/7.08 0.668/7.11 0.740/6.98 0.206/1.76 0.246/3.55 0.461/3.76 
Epoch [4/60], LTrain:0.074113 ATrain: 97.64% LTest:0.038229 ATest: 98.82% Time[s]: 21.50 w_clip/entropy[bits]: 0.690/7.14 0.697/7.07 0.758/7.00 0.226/1.76 0.268/3.53 0.488/3.77 
Epoch [5/60], LTrain:0.068885 ATrain: 97.84% LTest:0.033097 ATest: 98.86% Time[s]: 22.95 w_clip/entropy[bits]: 0.708/7.11 0.713/7.09 0.809/6.95 0.245/1.77 0.298/3.49 0.534/3.73 
Epoch [6/60], LTrain:0.063613 ATrain: 98.04% LTest:0.042567 ATest: 98.60% Time[s]: 24.58 w_clip/entropy[bits]: 0.777/6.98 0.789/6.96 0.818/6.91 0.264/1.76 0.315/3.50 0.570/3.69 
Epoch [7/60], LTrain:0.060857 ATrain: 98.08% LTest:0.035131 ATest: 98.92% Time[s]: 23.10 w_clip/entropy[bits]: 0.794/6.99 0.823/6.97 0.802/7.06 0.282/1.76 0.340/3.48 0.598/3.67 
Epoch [8/60], LTrain:0.056646 ATrain: 98.23% LTest:0.028163 ATest: 99.03% Time[s]: 23.28 w_clip/entropy[bits]: 0.845/6.92 0.816/7.00 0.830/7.00 0.300/1.75 0.356/3.49 0.624/3.68 
Epoch [9/60], LTrain:0.055925 ATrain: 98.22% LTest:0.027904 ATest: 99.14% Time[s]: 22.22 w_clip/entropy[bits]: 0.856/6.99 0.856/6.94 0.819/7.00 0.318/1.75 0.372/3.49 0.657/3.65 
Epoch [10/60], LTrain:0.051715 ATrain: 98.38% LTest:0.032421 ATest: 98.87% Time[s]: 19.61 w_clip/entropy[bits]: 0.901/6.89 0.952/6.86 0.852/6.99 0.332/1.75 0.396/3.47 0.679/3.65 
Epoch [11/60], LTrain:0.051969 ATrain: 98.36% LTest:0.028341 ATest: 99.07% Time[s]: 18.37 w_clip/entropy[bits]: 0.921/6.86 0.977/6.86 0.871/6.98 0.348/1.75 0.412/3.46 0.699/3.63 
Epoch [12/60], LTrain:0.049817 ATrain: 98.46% LTest:0.029459 ATest: 99.04% Time[s]: 18.53 w_clip/entropy[bits]: 0.926/6.90 1.007/6.83 0.853/7.01 0.362/1.75 0.422/3.48 0.722/3.62 
Epoch [13/60], LTrain:0.047917 ATrain: 98.49% LTest:0.025880 ATest: 99.17% Time[s]: 19.15 w_clip/entropy[bits]: 0.924/6.95 1.079/6.76 0.843/7.06 0.377/1.75 0.438/3.47 0.745/3.62 
Epoch [14/60], LTrain:0.045751 ATrain: 98.53% LTest:0.028785 ATest: 99.13% Time[s]: 19.21 w_clip/entropy[bits]: 0.935/6.91 1.112/6.72 0.932/6.97 0.391/1.75 0.451/3.47 0.772/3.59 
Epoch [15/60], LTrain:0.045239 ATrain: 98.56% LTest:0.022340 ATest: 99.23% Time[s]: 18.87 w_clip/entropy[bits]: 0.960/6.89 1.133/6.75 0.958/6.97 0.404/1.75 0.462/3.47 0.806/3.57 
Epoch [16/60], LTrain:0.043792 ATrain: 98.63% LTest:0.023063 ATest: 99.20% Time[s]: 19.60 w_clip/entropy[bits]: 0.902/7.00 1.173/6.71 1.001/6.91 0.417/1.74 0.472/3.48 0.909/3.43 
Epoch [17/60], LTrain:0.043388 ATrain: 98.62% LTest:0.020957 ATest: 99.37% Time[s]: 18.95 w_clip/entropy[bits]: 0.869/7.06 1.172/6.73 1.020/6.90 0.430/1.74 0.482/3.48 0.997/3.33 
Epoch [18/60], LTrain:0.040735 ATrain: 98.68% LTest:0.027036 ATest: 99.13% Time[s]: 20.31 w_clip/entropy[bits]: 0.864/7.09 1.205/6.69 1.021/6.91 0.441/1.74 0.494/3.47 1.056/3.28 
Epoch [19/60], LTrain:0.041058 ATrain: 98.67% LTest:0.024321 ATest: 99.30% Time[s]: 22.64 w_clip/entropy[bits]: 0.920/7.00 1.261/6.67 1.078/6.85 0.454/1.74 0.502/3.47 1.126/3.22 
Epoch [20/60], LTrain:0.038483 ATrain: 98.77% LTest:0.022144 ATest: 99.32% Time[s]: 20.92 w_clip/entropy[bits]: 0.921/7.03 1.323/6.62 1.120/6.80 0.465/1.73 0.510/3.47 1.170/3.18 
Epoch [21/60], LTrain:0.039801 ATrain: 98.75% LTest:0.021119 ATest: 99.34% Time[s]: 20.91 w_clip/entropy[bits]: 0.912/7.02 1.404/6.56 1.102/6.84 0.476/1.73 0.518/3.47 1.188/3.18 
Epoch [22/60], LTrain:0.038029 ATrain: 98.83% LTest:0.021477 ATest: 99.30% Time[s]: 21.19 w_clip/entropy[bits]: 0.924/7.03 1.372/6.60 1.118/6.79 0.485/1.74 0.526/3.47 1.217/3.15 
Epoch [23/60], LTrain:0.037371 ATrain: 98.79% LTest:0.020470 ATest: 99.31% Time[s]: 25.39 w_clip/entropy[bits]: 0.961/7.00 1.357/6.63 1.128/6.83 0.495/1.73 0.539/3.46 1.252/3.12 
Epoch [24/60], LTrain:0.035351 ATrain: 98.89% LTest:0.024447 ATest: 99.18% Time[s]: 21.92 w_clip/entropy[bits]: 0.948/7.00 1.400/6.59 1.167/6.79 0.503/1.74 0.544/3.46 1.276/3.11 
Epoch [25/60], LTrain:0.036360 ATrain: 98.86% LTest:0.021620 ATest: 99.30% Time[s]: 25.13 w_clip/entropy[bits]: 0.925/7.03 1.443/6.58 1.189/6.76 0.512/1.73 0.551/3.47 1.317/3.08 
Epoch [26/60], LTrain:0.036620 ATrain: 98.84% LTest:0.020044 ATest: 99.40% Time[s]: 24.69 w_clip/entropy[bits]: 0.942/7.05 1.440/6.54 1.175/6.78 0.520/1.73 0.555/3.46 1.358/3.05 
Epoch [27/60], LTrain:0.034659 ATrain: 98.90% LTest:0.021707 ATest: 99.27% Time[s]: 24.15 w_clip/entropy[bits]: 0.914/7.07 1.430/6.57 1.183/6.79 0.526/1.73 0.561/3.47 1.379/3.03 
Epoch [28/60], LTrain:0.034221 ATrain: 98.93% LTest:0.020409 ATest: 99.38% Time[s]: 24.84 w_clip/entropy[bits]: 0.950/7.05 1.432/6.59 1.202/6.75 0.533/1.73 0.561/3.47 1.402/3.05 
Epoch [29/60], LTrain:0.033031 ATrain: 98.95% LTest:0.019548 ATest: 99.41% Time[s]: 24.82 w_clip/entropy[bits]: 0.954/7.05 1.447/6.59 1.199/6.77 0.539/1.73 0.568/3.46 1.403/3.07 
Epoch [30/60], LTrain:0.031822 ATrain: 98.98% LTest:0.020747 ATest: 99.36% Time[s]: 25.48 w_clip/entropy[bits]: 0.934/7.03 1.473/6.56 1.208/6.77 0.544/1.73 0.575/3.46 1.413/3.06 
Epoch [31/60], LTrain:0.032101 ATrain: 98.97% LTest:0.019598 ATest: 99.38% Time[s]: 25.37 w_clip/entropy[bits]: 0.947/7.04 1.507/6.53 1.189/6.76 0.550/1.73 0.580/3.45 1.438/3.03 
Epoch [32/60], LTrain:0.031859 ATrain: 99.02% LTest:0.019532 ATest: 99.41% Time[s]: 25.29 w_clip/entropy[bits]: 0.948/7.03 1.557/6.50 1.188/6.80 0.554/1.73 0.586/3.45 1.453/3.05 
Epoch [33/60], LTrain:0.031497 ATrain: 98.99% LTest:0.019278 ATest: 99.41% Time[s]: 25.48 w_clip/entropy[bits]: 0.944/7.04 1.575/6.52 1.206/6.79 0.559/1.73 0.589/3.45 1.460/3.06 
Epoch [34/60], LTrain:0.029945 ATrain: 99.06% LTest:0.021466 ATest: 99.32% Time[s]: 25.52 w_clip/entropy[bits]: 0.955/7.08 1.574/6.49 1.234/6.78 0.563/1.73 0.594/3.44 1.464/3.07 
Epoch [35/60], LTrain:0.030311 ATrain: 99.07% LTest:0.018872 ATest: 99.51% Time[s]: 25.33 w_clip/entropy[bits]: 0.967/7.01 1.588/6.53 1.229/6.76 0.567/1.73 0.598/3.45 1.491/3.02 
Epoch [36/60], LTrain:0.028626 ATrain: 99.10% LTest:0.020642 ATest: 99.38% Time[s]: 25.28 w_clip/entropy[bits]: 0.974/7.08 1.575/6.50 1.228/6.75 0.571/1.73 0.600/3.45 1.519/3.00 
Epoch [37/60], LTrain:0.027847 ATrain: 99.13% LTest:0.019109 ATest: 99.44% Time[s]: 25.38 w_clip/entropy[bits]: 0.979/7.04 1.562/6.51 1.218/6.79 0.574/1.73 0.599/3.45 1.540/2.98 
Epoch [38/60], LTrain:0.027579 ATrain: 99.16% LTest:0.018988 ATest: 99.47% Time[s]: 25.17 w_clip/entropy[bits]: 1.000/6.98 1.567/6.51 1.239/6.78 0.577/1.73 0.601/3.45 1.552/2.99 
Epoch [39/60], LTrain:0.027095 ATrain: 99.15% LTest:0.019785 ATest: 99.39% Time[s]: 25.38 w_clip/entropy[bits]: 0.996/7.02 1.561/6.50 1.241/6.76 0.579/1.73 0.605/3.45 1.554/3.01 
Epoch [40/60], LTrain:0.026628 ATrain: 99.15% LTest:0.017582 ATest: 99.54% Time[s]: 25.10 w_clip/entropy[bits]: 1.003/7.01 1.567/6.50 1.246/6.75 0.581/1.73 0.607/3.44 1.556/3.01 
Epoch [41/60], LTrain:0.026357 ATrain: 99.18% LTest:0.019496 ATest: 99.45% Time[s]: 25.25 w_clip/entropy[bits]: 1.015/6.99 1.568/6.52 1.255/6.77 0.583/1.73 0.609/3.44 1.561/3.00 
Epoch [42/60], LTrain:0.025284 ATrain: 99.21% LTest:0.018024 ATest: 99.48% Time[s]: 25.22 w_clip/entropy[bits]: 1.025/7.02 1.580/6.46 1.258/6.71 0.584/1.73 0.611/3.44 1.571/3.00 
Epoch [43/60], LTrain:0.024707 ATrain: 99.22% LTest:0.017478 ATest: 99.47% Time[s]: 25.22 w_clip/entropy[bits]: 1.035/7.01 1.581/6.50 1.255/6.75 0.586/1.72 0.613/3.44 1.580/2.99 
Epoch [44/60], LTrain:0.024483 ATrain: 99.23% LTest:0.019240 ATest: 99.35% Time[s]: 25.22 w_clip/entropy[bits]: 1.042/6.97 1.585/6.51 1.270/6.74 0.587/1.73 0.615/3.43 1.587/3.00 
Epoch [45/60], LTrain:0.024742 ATrain: 99.23% LTest:0.018928 ATest: 99.43% Time[s]: 25.32 w_clip/entropy[bits]: 1.027/6.99 1.587/6.48 1.273/6.72 0.589/1.72 0.615/3.43 1.587/3.01 
Epoch [46/60], LTrain:0.024214 ATrain: 99.26% LTest:0.019358 ATest: 99.47% Time[s]: 25.21 w_clip/entropy[bits]: 1.020/7.01 1.597/6.48 1.281/6.74 0.589/1.73 0.615/3.43 1.591/3.00 
Epoch [47/60], LTrain:0.022988 ATrain: 99.27% LTest:0.019183 ATest: 99.41% Time[s]: 25.46 w_clip/entropy[bits]: 1.021/7.02 1.602/6.47 1.294/6.70 0.590/1.72 0.617/3.43 1.597/2.99 
Epoch [48/60], LTrain:0.023479 ATrain: 99.28% LTest:0.017610 ATest: 99.45% Time[s]: 25.01 w_clip/entropy[bits]: 1.028/7.00 1.604/6.46 1.292/6.69 0.591/1.73 0.617/3.43 1.604/2.99 
Epoch [49/60], LTrain:0.023090 ATrain: 99.27% LTest:0.016293 ATest: 99.49% Time[s]: 25.22 w_clip/entropy[bits]: 1.024/7.01 1.604/6.46 1.292/6.71 0.592/1.72 0.618/3.43 1.607/2.98 
Epoch [50/60], LTrain:0.022670 ATrain: 99.29% LTest:0.017558 ATest: 99.46% Time[s]: 25.19 w_clip/entropy[bits]: 1.031/6.99 1.604/6.47 1.296/6.68 0.592/1.72 0.619/3.43 1.610/2.99 
Epoch [51/60], LTrain:0.022138 ATrain: 99.33% LTest:0.017075 ATest: 99.52% Time[s]: 25.23 w_clip/entropy[bits]: 1.031/6.96 1.611/6.45 1.299/6.70 0.592/1.72 0.619/3.43 1.613/2.99 
Epoch [52/60], LTrain:0.021627 ATrain: 99.32% LTest:0.016306 ATest: 99.51% Time[s]: 25.01 w_clip/entropy[bits]: 1.029/6.97 1.613/6.45 1.303/6.71 0.593/1.73 0.619/3.43 1.615/2.98 
Epoch [53/60], LTrain:0.021072 ATrain: 99.34% LTest:0.016790 ATest: 99.55% Time[s]: 25.32 w_clip/entropy[bits]: 1.029/6.94 1.610/6.46 1.304/6.72 0.593/1.73 0.619/3.43 1.616/2.99 
Epoch [54/60], LTrain:0.021348 ATrain: 99.34% LTest:0.016587 ATest: 99.57% Time[s]: 25.33 w_clip/entropy[bits]: 1.030/6.96 1.613/6.47 1.303/6.70 0.593/1.73 0.619/3.43 1.618/2.98 
Epoch [55/60], LTrain:0.020614 ATrain: 99.37% LTest:0.016965 ATest: 99.53% Time[s]: 25.05 w_clip/entropy[bits]: 1.030/6.97 1.614/6.46 1.304/6.71 0.593/1.72 0.619/3.43 1.619/2.98 
Epoch [56/60], LTrain:0.020673 ATrain: 99.38% LTest:0.017001 ATest: 99.52% Time[s]: 25.27 w_clip/entropy[bits]: 1.029/6.97 1.614/6.48 1.305/6.73 0.593/1.73 0.619/3.43 1.620/2.98 
Epoch [57/60], LTrain:0.020005 ATrain: 99.37% LTest:0.016563 ATest: 99.56% Time[s]: 24.97 w_clip/entropy[bits]: 1.029/6.98 1.613/6.48 1.305/6.74 0.593/1.72 0.619/3.43 1.621/2.98 
Epoch [58/60], LTrain:0.021023 ATrain: 99.37% LTest:0.016283 ATest: 99.56% Time[s]: 25.12 w_clip/entropy[bits]: 1.029/6.97 1.613/6.48 1.305/6.73 0.593/1.72 0.619/3.43 1.622/2.98 
Epoch [59/60], LTrain:0.020598 ATrain: 99.38% LTest:0.016470 ATest: 99.55% Time[s]: 25.49 w_clip/entropy[bits]: 1.029/6.97 1.614/6.48 1.305/6.73 0.593/1.72 0.619/3.43 1.622/2.98 
Epoch [60/60], LTrain:0.020030 ATrain: 99.40% LTest:0.016605 ATest: 99.53% Time[s]: 25.16 w_clip/entropy[bits]: 1.029/6.97 1.614/6.49 1.305/6.73 0.593/1.73 0.619/3.43 1.622/2.98 
TotalBits: 90112 TotalBytes: 11264.0 

2bit mlp, 64 wide, 32groups 

Epoch [1/60], LTrain:0.273525 ATrain: 92.25% LTest:0.060230 ATest: 98.23% Time[s]: 20.05 Act: 44.1% w_clip/entropy[bits]: 0.463/7.43 0.419/7.35 0.451/7.28 0.117/1.89 0.177/3.74 0.416/3.58
Epoch [2/60], LTrain:0.105200 ATrain: 96.78% LTest:0.041287 ATest: 98.81% Time[s]: 24.16 Act: 43.7% w_clip/entropy[bits]: 0.503/7.37 0.497/7.22 0.493/7.23 0.170/1.82 0.205/3.68 0.488/3.54 
Epoch [3/60], LTrain:0.079418 ATrain: 97.57% LTest:0.037705 ATest: 98.87% Time[s]: 21.35 Act: 43.4% w_clip/entropy[bits]: 0.534/7.31 0.585/7.07 0.498/7.28 0.190/1.81 0.232/3.63 0.527/3.52 
Epoch [4/60], LTrain:0.069352 ATrain: 97.84% LTest:0.032338 ATest: 98.98% Time[s]: 22.79 Act: 43.0% w_clip/entropy[bits]: 0.527/7.40 0.670/6.93 0.509/7.30 0.214/1.80 0.255/3.61 0.551/3.54 
Epoch [5/60], LTrain:0.061584 ATrain: 98.08% LTest:0.037977 ATest: 98.90% Time[s]: 21.91 Act: 44.1% w_clip/entropy[bits]: 0.580/7.31 0.734/6.87 0.580/7.16 0.233/1.79 0.277/3.61 0.572/3.56 
Epoch [6/60], LTrain:0.057352 ATrain: 98.22% LTest:0.027144 ATest: 99.16% Time[s]: 21.79 Act: 42.7% w_clip/entropy[bits]: 0.603/7.27 0.820/6.76 0.600/7.16 0.251/1.79 0.297/3.60 0.611/3.53 
Epoch [7/60], LTrain:0.052723 ATrain: 98.32% LTest:0.028632 ATest: 99.14% Time[s]: 22.06 Act: 42.3% w_clip/entropy[bits]: 0.611/7.24 0.892/6.71 0.623/7.15 0.269/1.79 0.322/3.57 0.644/3.53
Epoch [8/60], LTrain:0.049972 ATrain: 98.42% LTest:0.025402 ATest: 99.27% Time[s]: 22.25 Act: 42.9% w_clip/entropy[bits]: 0.707/7.13 0.876/6.76 0.665/7.09 0.287/1.79 0.346/3.54 0.668/3.52
Epoch [9/60], LTrain:0.047577 ATrain: 98.53% LTest:0.028691 ATest: 99.03% Time[s]: 22.25 Act: 43.2% w_clip/entropy[bits]: 0.674/7.21 0.931/6.70 0.686/7.13 0.303/1.79 0.365/3.53 0.671/3.57
Epoch [10/60], LTrain:0.046486 ATrain: 98.52% LTest:0.028094 ATest: 99.11% Time[s]: 22.64 Act: 42.5% w_clip/entropy[bits]: 0.723/7.09 0.961/6.71 0.755/7.01 0.319/1.78 0.384/3.52 0.697/3.54
Epoch [11/60], LTrain:0.043646 ATrain: 98.62% LTest:0.030345 ATest: 98.97% Time[s]: 21.90 Act: 43.1% w_clip/entropy[bits]: 0.750/7.13 0.984/6.71 0.832/6.91 0.334/1.78 0.401/3.51 0.716/3.56
Epoch [12/60], LTrain:0.042982 ATrain: 98.63% LTest:0.023258 ATest: 99.35% Time[s]: 21.86 Act: 42.6% w_clip/entropy[bits]: 0.725/7.18 0.990/6.73 0.854/6.93 0.349/1.78 0.420/3.50 0.764/3.50
Epoch [13/60], LTrain:0.042072 ATrain: 98.67% LTest:0.024620 ATest: 99.20% Time[s]: 21.86 Act: 43.1% w_clip/entropy[bits]: 0.744/7.19 0.960/6.80 0.845/6.96 0.363/1.78 0.435/3.50 0.886/3.34
Epoch [14/60], LTrain:0.039839 ATrain: 98.76% LTest:0.024670 ATest: 99.29% Time[s]: 21.82 Act: 43.3% w_clip/entropy[bits]: 0.754/7.13 0.955/6.83 0.884/6.92 0.378/1.77 0.450/3.49 0.899/3.36
Epoch [15/60], LTrain:0.038174 ATrain: 98.82% LTest:0.021619 ATest: 99.32% Time[s]: 21.91 Act: 43.2% w_clip/entropy[bits]: 0.759/7.16 0.956/6.82 0.857/6.97 0.389/1.77 0.464/3.49 0.928/3.34
Epoch [16/60], LTrain:0.036203 ATrain: 98.85% LTest:0.020627 ATest: 99.27% Time[s]: 22.02 Act: 42.3% w_clip/entropy[bits]: 0.748/7.11 0.958/6.87 0.897/6.94 0.401/1.77 0.476/3.48 0.993/3.27
Epoch [17/60], LTrain:0.035542 ATrain: 98.88% LTest:0.023734 ATest: 99.26% Time[s]: 21.81 Act: 41.9% w_clip/entropy[bits]: 0.760/7.15 1.004/6.83 0.953/6.85 0.413/1.76 0.486/3.48 1.032/3.24
Epoch [18/60], LTrain:0.035026 ATrain: 98.90% LTest:0.026374 ATest: 99.19% Time[s]: 21.80 Act: 42.4% w_clip/entropy[bits]: 0.762/7.20 1.033/6.82 0.969/6.85 0.425/1.76 0.496/3.49 1.053/3.25
Epoch [19/60], LTrain:0.033073 ATrain: 98.96% LTest:0.024281 ATest: 99.33% Time[s]: 22.06 Act: 41.9% w_clip/entropy[bits]: 0.797/7.10 0.993/6.86 1.006/6.83 0.435/1.77 0.510/3.48 1.096/3.22
Epoch [20/60], LTrain:0.033751 ATrain: 98.92% LTest:0.024293 ATest: 99.27% Time[s]: 21.88 Act: 41.7% w_clip/entropy[bits]: 0.779/7.13 1.006/6.88 1.003/6.84 0.446/1.76 0.518/3.48 1.158/3.17
Epoch [21/60], LTrain:0.033608 ATrain: 98.95% LTest:0.026419 ATest: 99.15% Time[s]: 21.93 Act: 42.8% w_clip/entropy[bits]: 0.784/7.18 1.020/6.90 1.025/6.83 0.456/1.76 0.526/3.49 1.244/3.08
Epoch [22/60], LTrain:0.031409 ATrain: 99.00% LTest:0.026212 ATest: 99.26% Time[s]: 22.03 Act: 42.3% w_clip/entropy[bits]: 0.814/7.13 1.031/6.89 1.059/6.81 0.464/1.77 0.532/3.49 1.321/3.00
Epoch [23/60], LTrain:0.030354 ATrain: 99.03% LTest:0.024880 ATest: 99.33% Time[s]: 21.91 Act: 42.3% w_clip/entropy[bits]: 0.809/7.14 1.057/6.85 1.074/6.80 0.473/1.76 0.537/3.50 1.425/2.90
Epoch [24/60], LTrain:0.030384 ATrain: 99.04% LTest:0.023306 ATest: 99.38% Time[s]: 22.03 Act: 42.6% w_clip/entropy[bits]: 0.870/7.05 1.051/6.88 1.104/6.77 0.481/1.76 0.548/3.48 1.458/2.90
Epoch [25/60], LTrain:0.029590 ATrain: 99.07% LTest:0.021693 ATest: 99.32% Time[s]: 21.98 Act: 43.4% w_clip/entropy[bits]: 0.855/7.05 1.064/6.88 1.085/6.80 0.487/1.76 0.559/3.47 1.480/2.90
Epoch [26/60], LTrain:0.030110 ATrain: 99.08% LTest:0.024466 ATest: 99.25% Time[s]: 21.97 Act: 42.6% w_clip/entropy[bits]: 0.875/7.09 1.102/6.83 1.095/6.79 0.495/1.76 0.563/3.48 1.517/2.89
Epoch [27/60], LTrain:0.029005 ATrain: 99.06% LTest:0.020697 ATest: 99.32% Time[s]: 23.85 Act: 42.4% w_clip/entropy[bits]: 0.852/7.10 1.099/6.86 1.148/6.76 0.501/1.76 0.572/3.47 1.568/2.86
Epoch [28/60], LTrain:0.027700 ATrain: 99.10% LTest:0.025305 ATest: 99.26% Time[s]: 23.70 Act: 43.0% w_clip/entropy[bits]: 0.839/7.11 1.105/6.86 1.160/6.75 0.507/1.76 0.580/3.46 1.607/2.82
Epoch [29/60], LTrain:0.027948 ATrain: 99.13% LTest:0.021930 ATest: 99.34% Time[s]: 22.98 Act: 43.1% w_clip/entropy[bits]: 0.843/7.13 1.106/6.85 1.146/6.76 0.513/1.76 0.589/3.45 1.675/2.78
Epoch [30/60], LTrain:0.026948 ATrain: 99.16% LTest:0.020657 ATest: 99.32% Time[s]: 22.34 Act: 42.2% w_clip/entropy[bits]: 0.864/7.11 1.108/6.87 1.154/6.77 0.519/1.76 0.597/3.45 1.760/2.73
Epoch [31/60], LTrain:0.025668 ATrain: 99.18% LTest:0.021076 ATest: 99.43% Time[s]: 22.63 Act: 42.4% w_clip/entropy[bits]: 0.879/7.10 1.125/6.88 1.167/6.76 0.524/1.76 0.600/3.45 1.842/2.67
Epoch [32/60], LTrain:0.026964 ATrain: 99.15% LTest:0.019549 ATest: 99.40% Time[s]: 22.87 Act: 42.9% w_clip/entropy[bits]: 0.869/7.12 1.133/6.83 1.154/6.78 0.528/1.76 0.605/3.45 1.906/2.64
Epoch [33/60], LTrain:0.025921 ATrain: 99.20% LTest:0.025913 ATest: 99.21% Time[s]: 21.89 Act: 43.1% w_clip/entropy[bits]: 0.849/7.18 1.176/6.81 1.162/6.77 0.532/1.76 0.610/3.45 1.951/2.62
Epoch [34/60], LTrain:0.023480 ATrain: 99.26% LTest:0.021405 ATest: 99.39% Time[s]: 21.56 Act: 43.1% w_clip/entropy[bits]: 0.861/7.13 1.188/6.79 1.157/6.78 0.536/1.76 0.615/3.44 1.958/2.65
Epoch [35/60], LTrain:0.023950 ATrain: 99.23% LTest:0.022371 ATest: 99.34% Time[s]: 21.81 Act: 42.8% w_clip/entropy[bits]: 0.863/7.15 1.209/6.77 1.171/6.76 0.538/1.76 0.621/3.43 1.969/2.65
Epoch [36/60], LTrain:0.024283 ATrain: 99.26% LTest:0.023138 ATest: 99.33% Time[s]: 21.74 Act: 43.4% w_clip/entropy[bits]: 0.864/7.17 1.188/6.78 1.186/6.77 0.541/1.76 0.625/3.43 1.983/2.63
Epoch [37/60], LTrain:0.024043 ATrain: 99.25% LTest:0.023401 ATest: 99.32% Time[s]: 21.63 Act: 43.0% w_clip/entropy[bits]: 0.868/7.15 1.207/6.77 1.180/6.75 0.543/1.76 0.627/3.43 2.010/2.63
Epoch [45/60], LTrain:0.020375 ATrain: 99.38% LTest:0.020354 ATest: 99.36% Time[s]: 21.48 Act: 42.6% w_clip/entropy[bits]: 0.885/7.10 1.217/6.77 1.198/6.76 0.556/1.76 0.640/3.42 2.227/2.50      
Epoch [46/60], LTrain:0.019441 ATrain: 99.38% LTest:0.018541 ATest: 99.44% Time[s]: 21.62 Act: 42.6% w_clip/entropy[bits]: 0.879/7.14 1.218/6.78 1.203/6.74 0.557/1.76 0.642/3.42 2.234/2.50      
Epoch [47/60], LTrain:0.019504 ATrain: 99.39% LTest:0.017577 ATest: 99.41% Time[s]: 21.69 Act: 42.7% w_clip/entropy[bits]: 0.878/7.16 1.223/6.77 1.201/6.75 0.557/1.76 0.643/3.42 2.242/2.50      
Epoch [48/60], LTrain:0.018456 ATrain: 99.43% LTest:0.016583 ATest: 99.47% Time[s]: 21.46 Act: 42.7% w_clip/entropy[bits]: 0.880/7.13 1.228/6.76 1.205/6.74 0.558/1.76 0.643/3.42 2.249/2.49      
Epoch [49/60], LTrain:0.018686 ATrain: 99.42% LTest:0.018517 ATest: 99.44% Time[s]: 21.64 Act: 42.7% w_clip/entropy[bits]: 0.887/7.13 1.225/6.78 1.204/6.73 0.558/1.76 0.643/3.42 2.258/2.48      
Epoch [50/60], LTrain:0.018177 ATrain: 99.43% LTest:0.019077 ATest: 99.45% Time[s]: 21.56 Act: 42.6% w_clip/entropy[bits]: 0.882/7.15 1.231/6.76 1.203/6.74 0.559/1.76 0.644/3.42 2.262/2.48      
Epoch [51/60], LTrain:0.018281 ATrain: 99.44% LTest:0.018007 ATest: 99.45% Time[s]: 21.84 Act: 42.7% w_clip/entropy[bits]: 0.883/7.14 1.229/6.77 1.197/6.74 0.559/1.75 0.644/3.42 2.264/2.50      
Epoch [52/60], LTrain:0.018547 ATrain: 99.44% LTest:0.018670 ATest: 99.45% Time[s]: 21.87 Act: 42.7% w_clip/entropy[bits]: 0.886/7.12 1.232/6.76 1.201/6.74 0.559/1.75 0.644/3.42 2.266/2.50      
Epoch [53/60], LTrain:0.017670 ATrain: 99.44% LTest:0.018805 ATest: 99.43% Time[s]: 19.19 Act: 42.7% w_clip/entropy[bits]: 0.886/7.12 1.233/6.75 1.202/6.74 0.559/1.75 0.644/3.42 2.270/2.49      
Epoch [54/60], LTrain:0.017283 ATrain: 99.45% LTest:0.017706 ATest: 99.44% Time[s]: 19.08 Act: 42.7% w_clip/entropy[bits]: 0.888/7.14 1.233/6.76 1.202/6.74 0.559/1.75 0.644/3.42 2.273/2.49      
Epoch [55/60], LTrain:0.016962 ATrain: 99.48% LTest:0.017886 ATest: 99.44% Time[s]: 19.61 Act: 42.7% w_clip/entropy[bits]: 0.888/7.14 1.233/6.76 1.203/6.74 0.560/1.75 0.644/3.42 2.274/2.49      
Epoch [56/60], LTrain:0.016163 ATrain: 99.50% LTest:0.017486 ATest: 99.49% Time[s]: 19.16 Act: 42.7% w_clip/entropy[bits]: 0.888/7.11 1.234/6.76 1.203/6.73 0.560/1.75 0.644/3.42 2.275/2.49      
Epoch [57/60], LTrain:0.016734 ATrain: 99.49% LTest:0.017488 ATest: 99.51% Time[s]: 19.12 Act: 42.7% w_clip/entropy[bits]: 0.888/7.12 1.233/6.76 1.203/6.72 0.560/1.76 0.644/3.42 2.275/2.49      
Epoch [58/60], LTrain:0.015859 ATrain: 99.50% LTest:0.017606 ATest: 99.50% Time[s]: 19.19 Act: 42.7% w_clip/entropy[bits]: 0.888/7.12 1.233/6.76 1.203/6.72 0.560/1.75 0.644/3.42 2.275/2.49      
Epoch [59/60], LTrain:0.016334 ATrain: 99.47% LTest:0.017491 ATest: 99.50% Time[s]: 23.31 Act: 42.7% w_clip/entropy[bits]: 0.888/7.12 1.233/6.76 1.203/6.72 0.560/1.76 0.644/3.42 2.276/2.49 
Epoch [60/60], LTrain:0.015753 ATrain: 99.52% LTest:0.017618 ATest: 99.49% Time[s]: 24.98 Act: 42.7% w_clip/entropy[bits]: 0.888/7.12 1.233/6.76 1.203/6.72 0.560/1.76 0.644/3.42 2.276/2.49 
TotalBits: 99328 TotalBytes: 12416.0 
saving model...

2bit mlp, 64 wide, 16 groups
Epoch [1/60], LTrain:0.269279 ATrain: 92.32% LTest:0.059375 ATest: 98.26% Time[s]: 24.33 Act: 43.8% w_clip/entropy[bits]: 0.457/7.44 0.349/7.26 0.406/7.09 0.114/1.89 0.175/3.76 0.363/3.77 
Epoch [2/60], LTrain:0.098340 ATrain: 96.97% LTest:0.035143 ATest: 98.84% Time[s]: 25.56 Act: 43.0% w_clip/entropy[bits]: 0.506/7.38 0.391/7.20 0.457/7.01 0.162/1.82 0.204/3.68 0.410/3.77 
Epoch [3/60], LTrain:0.076069 ATrain: 97.66% LTest:0.032445 ATest: 99.06% Time[s]: 25.44 Act: 42.3% w_clip/entropy[bits]: 0.522/7.35 0.431/7.16 0.466/7.08 0.182/1.82 0.228/3.65 0.433/3.81 
Epoch [4/60], LTrain:0.064147 ATrain: 98.00% LTest:0.031866 ATest: 99.06% Time[s]: 25.33 Act: 41.6% w_clip/entropy[bits]: 0.577/7.28 0.459/7.15 0.501/7.05 0.202/1.81 0.250/3.64 0.468/3.79 
Epoch [5/60], LTrain:0.057827 ATrain: 98.24% LTest:0.029451 ATest: 99.01% Time[s]: 25.28 Act: 41.8% w_clip/entropy[bits]: 0.610/7.17 0.490/7.14 0.492/7.16 0.221/1.81 0.274/3.61 0.502/3.77 
Epoch [6/60], LTrain:0.053995 ATrain: 98.33% LTest:0.032041 ATest: 99.04% Time[s]: 25.21 Act: 40.4% w_clip/entropy[bits]: 0.600/7.25 0.533/7.10 0.555/7.05 0.239/1.80 0.295/3.60 0.557/3.70 
Epoch [7/60], LTrain:0.050881 ATrain: 98.44% LTest:0.033054 ATest: 98.93% Time[s]: 26.04 Act: 41.6% w_clip/entropy[bits]: 0.655/7.17 0.572/7.05 0.578/7.07 0.257/1.80 0.315/3.59 0.590/3.67 
Epoch [8/60], LTrain:0.046691 ATrain: 98.55% LTest:0.031144 ATest: 99.02% Time[s]: 24.47 Act: 41.1% w_clip/entropy[bits]: 0.658/7.17 0.650/6.94 0.604/7.06 0.275/1.79 0.335/3.57 0.622/3.64 
Epoch [9/60], LTrain:0.044277 ATrain: 98.66% LTest:0.028626 ATest: 99.15% Time[s]: 25.88 Act: 42.1% w_clip/entropy[bits]: 0.643/7.19 0.678/6.92 0.656/7.00 0.290/1.80 0.352/3.57 0.630/3.65 
Epoch [10/60], LTrain:0.042684 ATrain: 98.69% LTest:0.028422 ATest: 99.13% Time[s]: 24.46 Act: 40.4% w_clip/entropy[bits]: 0.645/7.25 0.787/6.78 0.654/7.05 0.305/1.80 0.373/3.55 0.653/3.61 
Epoch [11/60], LTrain:0.040245 ATrain: 98.72% LTest:0.026073 ATest: 99.26% Time[s]: 25.61 Act: 39.7% w_clip/entropy[bits]: 0.663/7.28 0.805/6.78 0.728/6.95 0.320/1.80 0.385/3.56 0.665/3.63 
Epoch [12/60], LTrain:0.039176 ATrain: 98.77% LTest:0.026877 ATest: 99.28% Time[s]: 25.03 Act: 40.6% w_clip/entropy[bits]: 0.703/7.17 0.790/6.86 0.786/6.88 0.333/1.80 0.401/3.55 0.696/3.60 
Epoch [13/60], LTrain:0.037648 ATrain: 98.84% LTest:0.025965 ATest: 99.17% Time[s]: 26.51 Act: 41.0% w_clip/entropy[bits]: 0.705/7.20 0.793/6.89 0.830/6.82 0.347/1.79 0.418/3.53 0.735/3.56 
Epoch [14/60], LTrain:0.035475 ATrain: 98.91% LTest:0.020008 ATest: 99.43% Time[s]: 25.41 Act: 39.9% w_clip/entropy[bits]: 0.735/7.13 0.859/6.82 0.841/6.85 0.359/1.79 0.429/3.54 0.756/3.54 
Epoch [15/60], LTrain:0.035943 ATrain: 98.88% LTest:0.022798 ATest: 99.32% Time[s]: 25.45 Act: 40.8% w_clip/entropy[bits]: 0.732/7.21 0.871/6.83 0.882/6.82 0.370/1.80 0.442/3.53 0.780/3.52 
Epoch [16/60], LTrain:0.034256 ATrain: 98.93% LTest:0.022956 ATest: 99.28% Time[s]: 25.78 Act: 40.7% w_clip/entropy[bits]: 0.713/7.21 0.834/6.91 0.889/6.83 0.382/1.79 0.453/3.53 0.800/3.52 
Epoch [17/60], LTrain:0.032555 ATrain: 98.95% LTest:0.023857 ATest: 99.25% Time[s]: 26.27 Act: 40.1% w_clip/entropy[bits]: 0.747/7.19 0.875/6.88 0.953/6.76 0.394/1.79 0.468/3.51 0.814/3.52 
Epoch [18/60], LTrain:0.032119 ATrain: 99.01% LTest:0.024356 ATest: 99.22% Time[s]: 25.49 Act: 39.8% w_clip/entropy[bits]: 0.726/7.26 0.878/6.87 0.960/6.78 0.405/1.79 0.474/3.53 0.814/3.54 
Epoch [19/60], LTrain:0.029840 ATrain: 99.03% LTest:0.020973 ATest: 99.37% Time[s]: 25.70 Act: 39.9% w_clip/entropy[bits]: 0.754/7.17 0.871/6.92 0.945/6.81 0.414/1.79 0.484/3.52 0.846/3.52 
Epoch [20/60], LTrain:0.029547 ATrain: 99.08% LTest:0.022521 ATest: 99.31% Time[s]: 26.24 Act: 39.6% w_clip/entropy[bits]: 0.741/7.23 0.905/6.90 0.964/6.81 0.424/1.79 0.490/3.53 0.874/3.50 
Epoch [21/60], LTrain:0.030468 ATrain: 99.06% LTest:0.027751 ATest: 99.22% Time[s]: 25.47 Act: 40.1% w_clip/entropy[bits]: 0.744/7.25 0.923/6.88 0.966/6.82 0.433/1.78 0.500/3.52 0.909/3.45 
Epoch [22/60], LTrain:0.028343 ATrain: 99.11% LTest:0.024082 ATest: 99.26% Time[s]: 24.83 Act: 40.2% w_clip/entropy[bits]: 0.767/7.23 0.972/6.82 0.992/6.80 0.442/1.78 0.511/3.52 0.949/3.40 
Epoch [23/60], LTrain:0.029168 ATrain: 99.09% LTest:0.022666 ATest: 99.33% Time[s]: 24.78 Act: 39.7% w_clip/entropy[bits]: 0.777/7.22 0.979/6.84 1.022/6.77 0.449/1.78 0.520/3.51 0.966/3.40 
Epoch [24/60], LTrain:0.027934 ATrain: 99.14% LTest:0.021162 ATest: 99.41% Time[s]: 26.02 Act: 40.0% w_clip/entropy[bits]: 0.763/7.23 1.024/6.79 1.051/6.74 0.456/1.78 0.530/3.50 0.985/3.38 
Epoch [25/60], LTrain:0.026325 ATrain: 99.14% LTest:0.023898 ATest: 99.28% Time[s]: 25.25 Act: 39.8% w_clip/entropy[bits]: 0.770/7.19 1.040/6.78 1.071/6.74 0.463/1.78 0.539/3.50 1.020/3.36 
Epoch [26/60], LTrain:0.027008 ATrain: 99.13% LTest:0.019793 ATest: 99.33% Time[s]: 25.15 Act: 39.7% w_clip/entropy[bits]: 0.792/7.21 1.060/6.77 1.096/6.71 0.469/1.79 0.543/3.50 1.042/3.34 
Epoch [27/60], LTrain:0.025417 ATrain: 99.22% LTest:0.022236 ATest: 99.37% Time[s]: 24.95 Act: 39.5% w_clip/entropy[bits]: 0.801/7.16 1.068/6.78 1.101/6.71 0.475/1.78 0.549/3.50 1.087/3.30 
Epoch [28/60], LTrain:0.024982 ATrain: 99.20% LTest:0.020888 ATest: 99.39% Time[s]: 26.18 Act: 40.6% w_clip/entropy[bits]: 0.782/7.20 1.088/6.75 1.109/6.72 0.481/1.78 0.557/3.50 1.127/3.28 
Epoch [29/60], LTrain:0.024120 ATrain: 99.24% LTest:0.018896 ATest: 99.47% Time[s]: 24.09 Act: 40.4% w_clip/entropy[bits]: 0.802/7.19 1.121/6.72 1.104/6.74 0.486/1.79 0.562/3.50 1.148/3.25 
Epoch [30/60], LTrain:0.023718 ATrain: 99.24% LTest:0.020925 ATest: 99.38% Time[s]: 24.34 Act: 39.7% w_clip/entropy[bits]: 0.812/7.20 1.142/6.71 1.099/6.76 0.490/1.78 0.570/3.49 1.167/3.24 
Epoch [31/60], LTrain:0.023016 ATrain: 99.29% LTest:0.020033 ATest: 99.41% Time[s]: 25.70 Act: 39.5% w_clip/entropy[bits]: 0.832/7.15 1.155/6.70 1.056/6.82 0.495/1.78 0.573/3.49 1.176/3.23 
Epoch [32/60], LTrain:0.023625 ATrain: 99.26% LTest:0.020646 ATest: 99.44% Time[s]: 25.93 Act: 40.1% w_clip/entropy[bits]: 0.837/7.14 1.142/6.72 1.048/6.83 0.499/1.78 0.576/3.49 1.190/3.21 
Epoch [33/60], LTrain:0.022533 ATrain: 99.30% LTest:0.021614 ATest: 99.40% Time[s]: 24.15 Act: 40.5% w_clip/entropy[bits]: 0.826/7.17 1.172/6.69 1.060/6.82 0.502/1.78 0.582/3.48 1.218/3.19 
Epoch [34/60], LTrain:0.021394 ATrain: 99.34% LTest:0.020514 ATest: 99.41% Time[s]: 24.40 Act: 40.1% w_clip/entropy[bits]: 0.825/7.22 1.149/6.72 1.062/6.83 0.505/1.78 0.586/3.48 1.245/3.18 
Epoch [35/60], LTrain:0.021459 ATrain: 99.32% LTest:0.019765 ATest: 99.47% Time[s]: 22.10 Act: 39.8% w_clip/entropy[bits]: 0.813/7.23 1.109/6.78 1.053/6.84 0.508/1.78 0.589/3.48 1.275/3.14 
Epoch [36/60], LTrain:0.021134 ATrain: 99.35% LTest:0.019551 ATest: 99.43% Time[s]: 23.50 Act: 40.1% w_clip/entropy[bits]: 0.833/7.21 1.119/6.76 1.076/6.82 0.511/1.79 0.591/3.48 1.296/3.13 
Epoch [37/60], LTrain:0.020473 ATrain: 99.37% LTest:0.020351 ATest: 99.42% Time[s]: 22.97 Act: 40.2% w_clip/entropy[bits]: 0.835/7.20 1.133/6.75 1.082/6.80 0.513/1.78 0.592/3.49 1.300/3.15 
Epoch [38/60], LTrain:0.019334 ATrain: 99.40% LTest:0.019622 ATest: 99.46% Time[s]: 23.61 Act: 40.2% w_clip/entropy[bits]: 0.837/7.14 1.151/6.73 1.086/6.81 0.516/1.78 0.592/3.49 1.313/3.14 
Epoch [39/60], LTrain:0.020049 ATrain: 99.38% LTest:0.020373 ATest: 99.43% Time[s]: 23.12 Act: 40.2% w_clip/entropy[bits]: 0.832/7.14 1.162/6.72 1.083/6.80 0.518/1.78 0.595/3.48 1.330/3.12 
Epoch [40/60], LTrain:0.018939 ATrain: 99.42% LTest:0.018541 ATest: 99.49% Time[s]: 22.17 Act: 40.7% w_clip/entropy[bits]: 0.822/7.21 1.179/6.71 1.093/6.80 0.520/1.78 0.595/3.49 1.352/3.11 
Epoch [41/60], LTrain:0.018870 ATrain: 99.42% LTest:0.019693 ATest: 99.45% Time[s]: 23.03 Act: 40.7% w_clip/entropy[bits]: 0.814/7.20 1.188/6.70 1.098/6.79 0.522/1.78 0.595/3.49 1.386/3.09 
Epoch [42/60], LTrain:0.018644 ATrain: 99.44% LTest:0.019467 ATest: 99.42% Time[s]: 22.01 Act: 40.6% w_clip/entropy[bits]: 0.820/7.19 1.186/6.70 1.098/6.79 0.523/1.78 0.598/3.49 1.405/3.08 
Epoch [43/60], LTrain:0.018130 ATrain: 99.47% LTest:0.018492 ATest: 99.47% Time[s]: 22.10 Act: 40.3% w_clip/entropy[bits]: 0.831/7.17 1.189/6.70 1.091/6.80 0.524/1.78 0.599/3.48 1.423/3.05 
Epoch [44/60], LTrain:0.017113 ATrain: 99.49% LTest:0.018650 ATest: 99.48% Time[s]: 21.99 Act: 40.4% w_clip/entropy[bits]: 0.832/7.19 1.199/6.69 1.090/6.81 0.525/1.78 0.602/3.48 1.435/3.04 
Epoch [45/60], LTrain:0.017323 ATrain: 99.47% LTest:0.016487 ATest: 99.56% Time[s]: 21.81 Act: 40.3% w_clip/entropy[bits]: 0.836/7.20 1.186/6.70 1.094/6.80 0.526/1.78 0.602/3.48 1.438/3.04 
Epoch [46/60], LTrain:0.017561 ATrain: 99.47% LTest:0.017670 ATest: 99.46% Time[s]: 23.09 Act: 40.0% w_clip/entropy[bits]: 0.830/7.17 1.192/6.69 1.084/6.81 0.526/1.78 0.603/3.48 1.439/3.04 
Epoch [47/60], LTrain:0.016723 ATrain: 99.46% LTest:0.019758 ATest: 99.44% Time[s]: 24.26 Act: 40.1% w_clip/entropy[bits]: 0.834/7.22 1.183/6.71 1.087/6.82 0.527/1.78 0.603/3.49 1.441/3.05 
Epoch [48/60], LTrain:0.015127 ATrain: 99.52% LTest:0.018027 ATest: 99.53% Time[s]: 23.02 Act: 40.0% w_clip/entropy[bits]: 0.836/7.21 1.186/6.71 1.087/6.81 0.527/1.78 0.604/3.48 1.442/3.05 
Epoch [49/60], LTrain:0.015221 ATrain: 99.53% LTest:0.017867 ATest: 99.55% Time[s]: 24.55 Act: 40.1% w_clip/entropy[bits]: 0.832/7.18 1.177/6.72 1.092/6.81 0.528/1.78 0.604/3.48 1.442/3.05 
Epoch [50/60], LTrain:0.015978 ATrain: 99.52% LTest:0.017262 ATest: 99.53% Time[s]: 23.65 Act: 39.9% w_clip/entropy[bits]: 0.831/7.21 1.176/6.72 1.095/6.81 0.528/1.78 0.604/3.48 1.444/3.06 
Epoch [51/60], LTrain:0.015425 ATrain: 99.51% LTest:0.016422 ATest: 99.53% Time[s]: 23.86 Act: 40.2% w_clip/entropy[bits]: 0.830/7.16 1.181/6.71 1.097/6.81 0.528/1.78 0.604/3.48 1.444/3.07 
Epoch [52/60], LTrain:0.015214 ATrain: 99.53% LTest:0.017965 ATest: 99.56% Time[s]: 23.93 Act: 40.5% w_clip/entropy[bits]: 0.832/7.14 1.183/6.71 1.099/6.80 0.529/1.78 0.605/3.48 1.445/3.06 
Epoch [53/60], LTrain:0.014316 ATrain: 99.57% LTest:0.017512 ATest: 99.52% Time[s]: 23.61 Act: 40.2% w_clip/entropy[bits]: 0.833/7.17 1.187/6.71 1.096/6.81 0.529/1.78 0.605/3.48 1.447/3.07 
Epoch [54/60], LTrain:0.014976 ATrain: 99.53% LTest:0.016840 ATest: 99.54% Time[s]: 23.83 Act: 40.2% w_clip/entropy[bits]: 0.832/7.20 1.189/6.71 1.094/6.81 0.529/1.78 0.605/3.48 1.448/3.06 
Epoch [55/60], LTrain:0.013901 ATrain: 99.58% LTest:0.016850 ATest: 99.52% Time[s]: 23.95 Act: 40.2% w_clip/entropy[bits]: 0.833/7.16 1.189/6.70 1.095/6.81 0.529/1.78 0.605/3.48 1.450/3.05 
Epoch [56/60], LTrain:0.013687 ATrain: 99.56% LTest:0.016479 ATest: 99.49% Time[s]: 23.85 Act: 40.2% w_clip/entropy[bits]: 0.833/7.15 1.189/6.70 1.095/6.81 0.529/1.78 0.605/3.48 1.452/3.05 
Epoch [57/60], LTrain:0.014033 ATrain: 99.57% LTest:0.016592 ATest: 99.55% Time[s]: 23.65 Act: 40.4% w_clip/entropy[bits]: 0.834/7.15 1.189/6.70 1.096/6.81 0.529/1.78 0.605/3.48 1.452/3.05 
Epoch [58/60], LTrain:0.013978 ATrain: 99.56% LTest:0.016925 ATest: 99.52% Time[s]: 23.57 Act: 40.3% w_clip/entropy[bits]: 0.834/7.14 1.189/6.70 1.096/6.81 0.529/1.78 0.605/3.48 1.452/3.05 
Epoch [59/60], LTrain:0.013760 ATrain: 99.57% LTest:0.016907 ATest: 99.52% Time[s]: 22.68 Act: 40.4% w_clip/entropy[bits]: 0.834/7.15 1.189/6.70 1.096/6.81 0.529/1.78 0.605/3.48 1.453/3.05 
Epoch [60/60], LTrain:0.013327 ATrain: 99.58% LTest:0.016684 ATest: 99.53% Time[s]: 22.31 Act: 40.4% w_clip/entropy[bits]: 0.834/7.15 1.189/6.70 1.096/6.81 0.529/1.78 0.605/3.48 1.453/3.05 
TotalBits: 117760 TotalBytes: 14720.0 

4bit mlp

training...
Epoch [1/60], LTrain:0.280140 ATrain: 92.02% LTest:0.073800 ATest: 97.80% Time[s]: 26.84 w_clip/entropy[bits]: 0.499/7.40 0.496/7.37 0.488/7.39 0.190/3.35 0.170/3.78 0.357/3.75 
Epoch [2/60], LTrain:0.107413 ATrain: 96.71% LTest:0.044789 ATest: 98.73% Time[s]: 24.21 w_clip/entropy[bits]: 0.613/7.17 0.548/7.32 0.568/7.23 0.239/3.22 0.209/3.63 0.405/3.74 
Epoch [3/60], LTrain:0.082888 ATrain: 97.45% LTest:0.039449 ATest: 98.77% Time[s]: 24.71 w_clip/entropy[bits]: 0.603/7.19 0.666/7.12 0.602/7.22 0.277/3.16 0.236/3.58 0.430/3.78 
Epoch [4/60], LTrain:0.073855 ATrain: 97.69% LTest:0.030664 ATest: 99.14% Time[s]: 24.38 w_clip/entropy[bits]: 0.583/7.28 0.690/7.15 0.649/7.17 0.312/3.13 0.263/3.53 0.466/3.75 
Epoch [5/60], LTrain:0.065570 ATrain: 97.99% LTest:0.030656 ATest: 99.11% Time[s]: 24.60 w_clip/entropy[bits]: 0.575/7.35 0.711/7.09 0.695/7.09 0.344/3.09 0.287/3.51 0.499/3.73 
Epoch [6/60], LTrain:0.060853 ATrain: 98.11% LTest:0.031569 ATest: 99.03% Time[s]: 24.20 w_clip/entropy[bits]: 0.627/7.31 0.780/7.01 0.728/7.06 0.377/3.07 0.306/3.49 0.523/3.72 
Epoch [7/60], LTrain:0.055810 ATrain: 98.25% LTest:0.030277 ATest: 99.09% Time[s]: 23.85 w_clip/entropy[bits]: 0.641/7.30 0.831/6.97 0.751/7.00 0.404/3.06 0.325/3.48 0.549/3.71 
Epoch [8/60], LTrain:0.053166 ATrain: 98.32% LTest:0.025180 ATest: 99.23% Time[s]: 23.55 w_clip/entropy[bits]: 0.662/7.26 0.985/6.81 0.784/7.06 0.432/3.04 0.343/3.47 0.582/3.68 
Epoch [9/60], LTrain:0.050875 ATrain: 98.43% LTest:0.023232 ATest: 99.26% Time[s]: 26.70 w_clip/entropy[bits]: 0.678/7.23 1.038/6.79 0.852/6.94 0.456/3.03 0.364/3.46 0.620/3.64 
Epoch [10/60], LTrain:0.050411 ATrain: 98.44% LTest:0.029316 ATest: 99.12% Time[s]: 25.87 w_clip/entropy[bits]: 0.684/7.31 1.071/6.79 0.865/6.94 0.481/3.02 0.375/3.46 0.685/3.56 
Epoch [11/60], LTrain:0.047499 ATrain: 98.54% LTest:0.027159 ATest: 99.15% Time[s]: 27.05 w_clip/entropy[bits]: 0.690/7.23 1.135/6.65 0.890/6.92 0.506/3.00 0.399/3.42 0.720/3.51 
Epoch [12/60], LTrain:0.044584 ATrain: 98.58% LTest:0.025023 ATest: 99.25% Time[s]: 26.43 w_clip/entropy[bits]: 0.678/7.28 1.153/6.70 0.996/6.76 0.526/2.99 0.416/3.41 0.766/3.48 
Epoch [13/60], LTrain:0.044200 ATrain: 98.58% LTest:0.025280 ATest: 99.19% Time[s]: 24.94 w_clip/entropy[bits]: 0.678/7.30 1.167/6.68 1.045/6.77 0.547/2.98 0.428/3.42 0.854/3.38 
Epoch [14/60], LTrain:0.041521 ATrain: 98.72% LTest:0.028101 ATest: 99.16% Time[s]: 26.12 w_clip/entropy[bits]: 0.680/7.31 1.172/6.72 1.039/6.77 0.565/2.98 0.441/3.41 0.971/3.22 
Epoch [15/60], LTrain:0.040717 ATrain: 98.72% LTest:0.027857 ATest: 99.13% Time[s]: 24.50 w_clip/entropy[bits]: 0.718/7.28 1.243/6.66 1.080/6.75 0.584/2.97 0.455/3.40 1.150/3.01 
Epoch [16/60], LTrain:0.038832 ATrain: 98.82% LTest:0.021329 ATest: 99.27% Time[s]: 26.28 w_clip/entropy[bits]: 0.739/7.23 1.314/6.58 1.057/6.80 0.602/2.96 0.465/3.41 1.325/2.87 
Epoch [17/60], LTrain:0.038318 ATrain: 98.82% LTest:0.022265 ATest: 99.32% Time[s]: 24.05 w_clip/entropy[bits]: 0.753/7.27 1.342/6.54 1.062/6.83 0.616/2.96 0.472/3.42 1.427/2.78 
Epoch [18/60], LTrain:0.037412 ATrain: 98.85% LTest:0.020843 ATest: 99.29% Time[s]: 23.51 w_clip/entropy[bits]: 0.778/7.20 1.307/6.61 1.063/6.81 0.631/2.95 0.477/3.43 1.626/2.63 
Epoch [19/60], LTrain:0.037285 ATrain: 98.84% LTest:0.021468 ATest: 99.42% Time[s]: 23.68 w_clip/entropy[bits]: 0.770/7.22 1.342/6.58 1.056/6.82 0.644/2.95 0.491/3.42 1.708/2.57 
Epoch [20/60], LTrain:0.034591 ATrain: 98.89% LTest:0.020531 ATest: 99.38% Time[s]: 23.38 w_clip/entropy[bits]: 0.752/7.27 1.353/6.60 1.080/6.82 0.658/2.94 0.499/3.41 1.844/2.51 
Epoch [21/60], LTrain:0.034750 ATrain: 98.92% LTest:0.020519 ATest: 99.39% Time[s]: 23.24 w_clip/entropy[bits]: 0.790/7.21 1.360/6.57 1.118/6.78 0.669/2.94 0.505/3.42 1.884/2.48 
Epoch [22/60], LTrain:0.034378 ATrain: 98.90% LTest:0.022741 ATest: 99.35% Time[s]: 23.73 w_clip/entropy[bits]: 0.804/7.29 1.397/6.58 1.083/6.87 0.683/2.94 0.520/3.40 1.908/2.50 
Epoch [23/60], LTrain:0.033335 ATrain: 98.98% LTest:0.024188 ATest: 99.34% Time[s]: 23.74 w_clip/entropy[bits]: 0.805/7.23 1.422/6.58 1.090/6.82 0.692/2.94 0.528/3.40 1.934/2.50 
Epoch [24/60], LTrain:0.033024 ATrain: 98.96% LTest:0.022000 ATest: 99.33% Time[s]: 23.89 w_clip/entropy[bits]: 0.825/7.21 1.400/6.56 1.126/6.82 0.704/2.94 0.537/3.39 1.973/2.47 
Epoch [25/60], LTrain:0.032755 ATrain: 98.95% LTest:0.019204 ATest: 99.33% Time[s]: 23.74 w_clip/entropy[bits]: 0.806/7.24 1.451/6.58 1.102/6.83 0.716/2.93 0.544/3.39 2.014/2.46 
Epoch [26/60], LTrain:0.030758 ATrain: 99.05% LTest:0.021182 ATest: 99.33% Time[s]: 23.96 w_clip/entropy[bits]: 0.813/7.23 1.497/6.53 1.118/6.80 0.724/2.92 0.550/3.39 2.052/2.46 
Epoch [27/60], LTrain:0.031116 ATrain: 99.03% LTest:0.019553 ATest: 99.28% Time[s]: 23.82 w_clip/entropy[bits]: 0.812/7.21 1.578/6.46 1.096/6.85 0.730/2.93 0.555/3.39 2.087/2.44 
Epoch [28/60], LTrain:0.029046 ATrain: 99.07% LTest:0.019106 ATest: 99.35% Time[s]: 23.74 w_clip/entropy[bits]: 0.830/7.19 1.625/6.40 1.140/6.84 0.738/2.93 0.559/3.39 2.149/2.41 
Epoch [29/60], LTrain:0.029817 ATrain: 99.05% LTest:0.018342 ATest: 99.39% Time[s]: 23.76 w_clip/entropy[bits]: 0.829/7.24 1.612/6.45 1.144/6.84 0.744/2.92 0.564/3.39 2.215/2.37 
Epoch [30/60], LTrain:0.027870 ATrain: 99.13% LTest:0.020384 ATest: 99.37% Time[s]: 23.82 w_clip/entropy[bits]: 0.839/7.24 1.638/6.41 1.151/6.80 0.749/2.93 0.568/3.38 2.221/2.42 
Epoch [31/60], LTrain:0.027983 ATrain: 99.10% LTest:0.021143 ATest: 99.29% Time[s]: 23.70 w_clip/entropy[bits]: 0.813/7.24 1.656/6.41 1.164/6.80 0.755/2.93 0.572/3.39 2.246/2.38 
Epoch [32/60], LTrain:0.026020 ATrain: 99.19% LTest:0.018749 ATest: 99.42% Time[s]: 25.38 w_clip/entropy[bits]: 0.814/7.26 1.650/6.42 1.142/6.78 0.762/2.92 0.577/3.38 2.275/2.36 
Epoch [33/60], LTrain:0.027312 ATrain: 99.15% LTest:0.020506 ATest: 99.35% Time[s]: 25.31 w_clip/entropy[bits]: 0.838/7.29 1.656/6.39 1.142/6.80 0.765/2.92 0.583/3.38 2.333/2.36 
Epoch [34/60], LTrain:0.026533 ATrain: 99.17% LTest:0.017750 ATest: 99.43% Time[s]: 25.06 w_clip/entropy[bits]: 0.841/7.25 1.647/6.42 1.138/6.81 0.771/2.92 0.588/3.37 2.391/2.33 
Epoch [35/60], LTrain:0.026137 ATrain: 99.20% LTest:0.018567 ATest: 99.45% Time[s]: 25.26 w_clip/entropy[bits]: 0.831/7.23 1.655/6.42 1.145/6.85 0.777/2.91 0.591/3.37 2.439/2.31 
Epoch [36/60], LTrain:0.026583 ATrain: 99.17% LTest:0.018160 ATest: 99.43% Time[s]: 25.20 w_clip/entropy[bits]: 0.850/7.21 1.642/6.42 1.159/6.84 0.781/2.91 0.592/3.38 2.495/2.31 
Epoch [37/60], LTrain:0.025586 ATrain: 99.19% LTest:0.017718 ATest: 99.41% Time[s]: 25.54 w_clip/entropy[bits]: 0.857/7.24 1.623/6.45 1.144/6.84 0.784/2.91 0.594/3.38 2.535/2.29 
Epoch [38/60], LTrain:0.024944 ATrain: 99.22% LTest:0.019502 ATest: 99.43% Time[s]: 25.19 w_clip/entropy[bits]: 0.856/7.21 1.625/6.46 1.148/6.84 0.787/2.91 0.596/3.38 2.566/2.28 
Epoch [39/60], LTrain:0.024757 ATrain: 99.23% LTest:0.022663 ATest: 99.29% Time[s]: 25.31 w_clip/entropy[bits]: 0.857/7.18 1.605/6.46 1.150/6.85 0.788/2.91 0.597/3.37 2.613/2.27 
Epoch [40/60], LTrain:0.025003 ATrain: 99.22% LTest:0.018734 ATest: 99.43% Time[s]: 25.11 w_clip/entropy[bits]: 0.876/7.18 1.609/6.46 1.158/6.81 0.789/2.91 0.598/3.38 2.628/2.26 
Epoch [41/60], LTrain:0.023419 ATrain: 99.28% LTest:0.018621 ATest: 99.46% Time[s]: 25.36 w_clip/entropy[bits]: 0.864/7.20 1.614/6.46 1.174/6.80 0.791/2.91 0.599/3.38 2.637/2.27 
Epoch [42/60], LTrain:0.023091 ATrain: 99.29% LTest:0.018323 ATest: 99.43% Time[s]: 25.29 w_clip/entropy[bits]: 0.869/7.19 1.623/6.43 1.162/6.80 0.792/2.91 0.599/3.38 2.654/2.26 
Epoch [43/60], LTrain:0.022368 ATrain: 99.29% LTest:0.018484 ATest: 99.38% Time[s]: 25.45 w_clip/entropy[bits]: 0.869/7.24 1.618/6.43 1.151/6.81 0.793/2.91 0.602/3.37 2.670/2.26 
Epoch [44/60], LTrain:0.023405 ATrain: 99.31% LTest:0.019622 ATest: 99.46% Time[s]: 25.37 w_clip/entropy[bits]: 0.869/7.16 1.630/6.42 1.163/6.79 0.795/2.91 0.603/3.37 2.687/2.26 
Epoch [45/60], LTrain:0.023481 ATrain: 99.28% LTest:0.018491 ATest: 99.46% Time[s]: 25.23 w_clip/entropy[bits]: 0.878/7.15 1.631/6.46 1.165/6.83 0.796/2.91 0.603/3.37 2.702/2.25 
Epoch [46/60], LTrain:0.022140 ATrain: 99.30% LTest:0.017929 ATest: 99.35% Time[s]: 25.62 w_clip/entropy[bits]: 0.878/7.19 1.633/6.44 1.166/6.80 0.797/2.91 0.603/3.37 2.717/2.25 
Epoch [47/60], LTrain:0.020767 ATrain: 99.36% LTest:0.019444 ATest: 99.42% Time[s]: 25.34 w_clip/entropy[bits]: 0.880/7.19 1.634/6.45 1.172/6.78 0.798/2.91 0.603/3.37 2.727/2.24 
Epoch [48/60], LTrain:0.021642 ATrain: 99.32% LTest:0.017317 ATest: 99.41% Time[s]: 25.25 w_clip/entropy[bits]: 0.889/7.22 1.633/6.46 1.166/6.80 0.798/2.91 0.602/3.38 2.737/2.23 
Epoch [49/60], LTrain:0.021045 ATrain: 99.35% LTest:0.016369 ATest: 99.48% Time[s]: 25.52 w_clip/entropy[bits]: 0.886/7.16 1.627/6.43 1.165/6.77 0.798/2.91 0.603/3.37 2.747/2.23 
Epoch [50/60], LTrain:0.020035 ATrain: 99.40% LTest:0.016755 ATest: 99.48% Time[s]: 25.25 w_clip/entropy[bits]: 0.883/7.17 1.627/6.43 1.171/6.81 0.798/2.91 0.603/3.37 2.751/2.24 
Epoch [51/60], LTrain:0.020385 ATrain: 99.37% LTest:0.017537 ATest: 99.49% Time[s]: 25.26 w_clip/entropy[bits]: 0.884/7.17 1.626/6.44 1.168/6.82 0.798/2.91 0.604/3.37 2.754/2.24 
Epoch [52/60], LTrain:0.019252 ATrain: 99.41% LTest:0.018219 ATest: 99.44% Time[s]: 25.00 w_clip/entropy[bits]: 0.889/7.16 1.624/6.44 1.166/6.82 0.799/2.91 0.604/3.37 2.758/2.24 
Epoch [53/60], LTrain:0.020902 ATrain: 99.34% LTest:0.017604 ATest: 99.43% Time[s]: 23.25 w_clip/entropy[bits]: 0.889/7.16 1.625/6.42 1.165/6.82 0.799/2.91 0.605/3.37 2.761/2.23 
Epoch [54/60], LTrain:0.018936 ATrain: 99.43% LTest:0.017970 ATest: 99.44% Time[s]: 24.14 w_clip/entropy[bits]: 0.889/7.14 1.625/6.44 1.165/6.79 0.799/2.91 0.605/3.37 2.764/2.23 
Epoch [55/60], LTrain:0.018598 ATrain: 99.42% LTest:0.017754 ATest: 99.43% Time[s]: 24.83 w_clip/entropy[bits]: 0.888/7.17 1.625/6.46 1.165/6.81 0.799/2.91 0.605/3.37 2.767/2.22 
Epoch [56/60], LTrain:0.020071 ATrain: 99.39% LTest:0.017203 ATest: 99.43% Time[s]: 25.10 w_clip/entropy[bits]: 0.888/7.16 1.624/6.45 1.165/6.79 0.799/2.91 0.605/3.37 2.768/2.22 
Epoch [57/60], LTrain:0.018656 ATrain: 99.42% LTest:0.017223 ATest: 99.46% Time[s]: 26.03 w_clip/entropy[bits]: 0.888/7.16 1.624/6.45 1.164/6.79 0.799/2.91 0.605/3.37 2.769/2.23 
Epoch [58/60], LTrain:0.017672 ATrain: 99.45% LTest:0.016865 ATest: 99.46% Time[s]: 27.06 w_clip/entropy[bits]: 0.888/7.15 1.624/6.45 1.164/6.79 0.799/2.91 0.605/3.37 2.770/2.23 
Epoch [59/60], LTrain:0.018378 ATrain: 99.44% LTest:0.017043 ATest: 99.46% Time[s]: 26.20 w_clip/entropy[bits]: 0.888/7.15 1.624/6.44 1.164/6.79 0.799/2.91 0.605/3.37 2.770/2.24 
Epoch [60/60], LTrain:0.019358 ATrain: 99.41% LTest:0.017571 ATest: 99.44% Time[s]: 26.37 w_clip/entropy[bits]: 0.888/7.15 1.624/6.44 1.164/6.78 0.799/2.91 0.605/3.37 2.770/2.23 


48 wide, 2 bit

Epoch [58/60], LTrain:0.022044 ATrain: 99.30% LTest:0.019007 ATest: 99.41% Time[s]: 18.50 w_clip/entropy[bits]: 1.080/6.82 1.612/6.43 1.166/6.70 0.659/1.75 0.658/3.40 1.347/3.19 
Epoch [59/60], LTrain:0.022665 ATrain: 99.29% LTest:0.018640 ATest: 99.42% Time[s]: 18.45 w_clip/entropy[bits]: 1.080/6.81 1.612/6.44 1.166/6.70 0.659/1.75 0.658/3.41 1.347/3.19 
Epoch [60/60], LTrain:0.021771 ATrain: 99.30% LTest:0.018820 ATest: 99.44% Time[s]: 18.65 w_clip/entropy[bits]: 1.080/6.82 1.612/6.44 1.166/6.69 0.659/1.75 0.658/3.41 1.347/3.19 
TotalBits: 74368 TotalBytes: 9296.0 
saving model...



32wide, 2 bit

Epoch [59/60], LTrain:0.028556 ATrain: 99.10% LTest:0.022346 ATest: 99.31% Time[s]: 22.82 Act: 45.5% w_clip/entropy[bits]: 1.139/6.66 1.716/6.28 0.938/6.67 0.732/1.76 0.649/3.51 1.048/3.53 
Epoch [60/60], LTrain:0.027292 ATrain: 99.12% LTest:0.022582 ATest: 99.28% Time[s]: 22.65 Act: 45.5% w_clip/entropy[bits]: 1.139/6.66 1.716/6.28 0.938/6.67 0.732/1.76 0.649/3.51 1.048/3.53 
TotalBits: 58624 TotalBytes: 7328.0 

64wide, 2 bit, 90ep

Total number of bits: 90112 (11.0 kbytes)
inference of quantized model
Accuracy/Test of quantized model: 99.53999999999999 %
Exporting model to header file

Epoch [89/90], LTrain:0.016856 ATrain: 99.46% LTest:0.015817 ATest: 99.54% Time[s]: 28.26 Act: 47.3% w_clip/entropy[bits]: 1.092/7.05 2.113/6.29 1.731/6.49 0.758/1.72 0.793/3.28 1.494/3.29 
Epoch [90/90], LTrain:0.017462 ATrain: 99.47% LTest:0.015791 ATest: 99.55% Time[s]: 27.89 Act: 47.3% w_clip/entropy[bits]: 1.092/7.05 2.113/6.29 1.731/6.49 0.758/1.72 0.793/3.28 1.494/3.29

80wide,2 bit, 90ep

Epoch [89/90], LTrain:0.014727 ATrain: 99.54% LTest:0.017152 ATest: 99.44% Time[s]: 28.64 Act: 45.2% w_clip/entropy[bits]: 1.117/7.07 1.442/6.83 1.776/6.48 0.676/1.72 0.738/3.32 1.332/3.40 
Epoch [90/90], LTrain:0.015139 ATrain: 99.51% LTest:0.017336 ATest: 99.42% Time[s]: 28.61 Act: 45.2% w_clip/entropy[bits]: 1.117/7.06 1.442/6.84 1.776/6.48 0.676/1.72 0.738/3.32 1.332/3.40 
TotalBits: 105856 TotalBytes: 13232.0 

16wide, 2 bit, 60ep
Epoch [60/60], LTrain:0.048836 ATrain: 98.43% LTest:0.029402 ATest: 99.06% Time[s]: 27.15 Act: 45.0% w_clip/entropy[bits]: 0.833/6.49 1.716/6.05 0.810/6.37 0.925/1.78 0.707/3.46 1.035/3.63 
TotalBits: 42880 TotalBytes: 5360.0 
Starting MNIST inference...
Inference of Sample 1   Prediction: 7   Label: 7        Timing: 784366 clock cycles
Inference of Sample 2   Prediction: 1   Label: 1        Timing: 787417 clock cycles
Inference of Sample 3   Prediction: 9   Label: 9        Timing: 783585 clock cycles


16wide, 4bit,4bit,60ep
Epoch [59/60], LTrain:0.042492 ATrain: 98.65% LTest:0.024877 ATest: 99.12% Time[s]: 27.19 Act: 50.2% w_clip/entropy[bits]: 0.885/6.43 1.319/6.18 0.752/6.49 1.105/2.96 0.708/3.42 1.161/3.50 
Epoch [60/60], LTrain:0.043009 ATrain: 98.63% LTest:0.024892 ATest: 99.09% Time[s]: 27.19 Act: 50.2% w_clip/entropy[bits]: 0.885/6.43 1.319/6.16 0.752/6.51 1.105/2.96 0.708/3.42 1.161/3.50 
TotalBits: 55168 TotalBytes: 6896.0 

Starting MNIST inference...
Inference of Sample 1   Prediction: 7   Label: 7        Timing: 740106 clock cycles
Inference of Sample 2   Prediction: 1   Label: 1        Timing: 740119 clock cycles
Inference of Sample 3   Prediction: 9   Label: 9        Timing: 739585 clock cycles


| Configuration | Width | Quantization | Epochs | Train Accuracy | Test Accuracy | Test Error | Model Size |
|---------------|-------|--------------|--------|----------------|---------------|------------|------------|
| 16-wide 2-bit | 16 | 2-bit | 60 | 98.43% | 99.06% | 0.94% | 42,880 bits (5.4 kb)
| 32-wide 2-bit | 32 | 2-bit | 60 | 99.12% | 99.28% | 0.72% | 58,624 bits (7.3 kB) |
| 48-wide 2-bit | 48 | 2-bit | 60 | 99.30% | 99.44% | 0.56% | 74,368 bits (9.3 kB) |
| 64-wide 2-bit MLP | 64 | 2-bit | 60 | 99.40% | *99.53%* | *0.47%* | 90,112 bits (11.0 kB) |
| 64-wide 4-bit MLP | 64 | 4-bit | 60 | 99.41% | 99.44% | 0.56% | 100,864 bits (12.3 kB) |
| 64-wide 2-bit (90ep) | 64 | 2-bit | 90 | 99.47% | **99.55%** | **0.45%** | 90,112 bits (11.0 kB) |
| 80-wide 2-bit (90ep) | 80 | 2-bit | 90 | 99.51% | 99.42% | 0.58% | 105,856 bits (13.2 kB) |



64-wide 2-bit (90ep) + 50% elastic
Epoch [90/90], LTrain:0.027318 ATrain: 99.14% LTest:0.015479 ATest: 99.51% Time[s]: 32.58 Act: 48.2% w_clip/entropy[bits]: 1.209/6.85 1.489/6.71 1.204/6.87 0.792/1.73 0.754/3.40 1.395/3.31 
TotalBits: 90112 TotalBytes: 11264.0 

# References

[^1]: S. Ma et al *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) and [discussion here](https://huggingface.co/papers/2402.17764) 

[^2]: A. Gholami et al. *A Survey of Quantization Methods for Efficient Neural Network Inference* ([arXiv:2103.13630](https://arxiv.org/abs/2103.13630))

[^3]: M. Rastegari et al. *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks* ([arXiv:1603.05279](https://arxiv.org/abs/1603.05279)) and *BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1* ([arXiv:1602.02830](https://arxiv.org/abs/1602.02830))

[^4]: S. Ma et al. *The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ* ([Github](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf))

[^5]: Y. Bengio et al. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation* [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)

[^6]: B. Zhang et al.  *Root Mean Square Layer Normalization* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

[^7]: M. Courbariaux et al. *BinaryConnect: Training Deep Neural Networks with binary weights during propagations* [arXiv:1511.00363](https://arxiv.org/abs/1511.00363)

[^8]: M. Elhoushi et al. *DeepShift: Towards Multiplication-Less Neural Networks*  [arXiv:1905.13298](https://arxiv.org/abs/1905.13298)

[^9]: C. Sakr et al. *Optimal Clipping and Magnitude-aware Differentiation for Improved Quantization-aware Training* [arXiv:2206.06501](https://arxiv.org/abs/2206.06501)

[^10]: T. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs* [[arXiv:2305.14314]](https://arxiv.org/pdf/2305.14314)


[^11]: Y. LeCun et al. *Gradient-based learning applied to document recognition* ([Proc. IEEE, 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf))

[^12]: D. Cireşan et al. *Multi-column Deep Neural Networks for Image Classification* ([arXiv:1202.2745](https://arxiv.org/abs/1202.2745))

[^13]: A. Krizhevsky et al. *ImageNet Classification with Deep Convolutional Neural Networks* ([Proceedings of the 25 th International Conference on Neural Information Processing Systems](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)), Dec 2012

[^14]: A. G. Howard et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* ([arXiv:1704.04861](https://arxiv.org/abs/1704.04861))

[^15]: F. Chollet et al. *Xception: Deep Learning with Depthwise Separable Convolutions* ([arXiv:1610.02357](https://arxiv.org/abs/1610.02357))

[^16]: J. Lin et al. *MCUNet: Tiny Deep Learning on IoT Devices* ([arXiv:2007.10319](https://arxiv.org/abs/2007.10319))

