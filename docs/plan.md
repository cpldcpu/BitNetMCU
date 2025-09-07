# Plan
- [Plan](#plan)
- [Funnel](#funnel)
  - [Self-compressing models](#self-compressing-models)
  - [Refactor models.py](#refactor-modelspy)
  - [Implement Conv2d](#implement-conv2d)
  - [Investigate network capacity scaling also with data augmentation](#investigate-network-capacity-scaling-also-with-data-augmentation)
  - [Implement pruning to optimize density](#implement-pruning-to-optimize-density)
  - [Lottery ticket search](#lottery-ticket-search)
  - [1.58 bit inference](#158-bit-inference)
  - [4.6 bit quantization](#46-bit-quantization)
- [Parking lot](#parking-lot)
  - [Optimize fc model](#optimize-fc-model)
  - [Stochastic weight averaging](#stochastic-weight-averaging)
  - [Regularization by switching quantization levels](#regularization-by-switching-quantization-levels)
  - [Distilliation](#distilliation)
- [Done](#done)
  - [Implement OCTAV](#implement-octav)
  - [Refactor code to introduce clipping parameter](#refactor-code-to-introduce-clipping-parameter)
  - [Implement two step learning schedule](#implement-two-step-learning-schedule)

---


# Funnel

## Ternary encoding

### packing 


```c
// Pack 20 trits (values in {-1,0,1}) into a 32-bit word
uint32_t pack_trits_32(const int8_t* trits) {
    uint32_t value = 0;
    
    // Build base-3 number from trits (least significant first)
    for (int i = 19; i >= 0; i--) {
        value = value * 3 + (trits[i] + 1);  // Map {-1,0,1} to {0,1,2}
    }
    
    // Scale so first trit appears in MSBs
    // Multiply by 2^32 / 3^20 using fixed-point arithmetic
    // 2^32 / 3^20 ≈ 1.23116 
    // Using 64-bit intermediate to avoid overflow
    uint64_t scaled = ((uint64_t)value << 32) / 3486784401ULL;
    
    // Ceiling division to avoid errors during unpacking
    if (((uint64_t)value << 32) % 3486784401ULL != 0) {
        scaled++;
    }
    
    return (uint32_t)scaled;
}
```
### unpacking
```c
// Unpack 20 trits from a 32-bit word
void unpack_trits_32(uint32_t packed, int8_t* trits) {
    uint32_t val = packed;
    
    for (int i = 0; i < 20; i++) {
        // Extract trit from the two MSBs
        uint8_t trit = val >> 30;
        trits[i] = trit - 1;  // Convert {0,1,2} to {-1,0,1}
        
        // Clear the MSBs and shift up remaining trits
        val = (val & 0x3FFFFFFF) * 3;
    }
}
```

### inference?

```c
   // encoding:
   // 0 = 00 = +1
   // 1 = 01 = -1
   // 2 = 10 =  0

      int32_t sum=0;
      for (uint32_t j = 0; j < 16; j++) {
          if !(weightChunk & 0x80000000) {
              int32_t in = *activations_idx;
              sum += (weightChunk & 0x40000000) ? -in : in; 
          }
          activations_idx++;
          weightChunk += (weightChunk << 1); // weightchunk = weightchunk * 3
      }
```


## Activation functions other than ReLU

Can activations functions help to improve network capacity and alleviate quantization noise?

- ReLU2 -> better expressability ???
- HardSwish -> Let negative activitations contribute to reduce noise?

width48_48_48_epochs30

Test

| Run         | Smoothed  | Step | Valuee | Training Time |
|-------------|----------------|------|----------|---------------|
| GELU        | 98.5673        | 30   | 98.6178  | 6.246 min     |
| LeakyReLU   | 98.4166        | 30   | 98.4876  | 6.196 min     |
| HardSwish   | 98.6103        | 30   | 98.728   | 6.269 min     |
| ReLU       | 98.3856        | 30   | 98.4275  | 6.264 min     |

Train
| Run       | Smoothed | Value  | Step | Training Time |
|-----------|----------|--------|------|---------------|
| GELU      | 97.501   | 97.6358| 30   | 6.246 min     |
| LeakyReLU | 97.3084  | 97.4075| 30   | 6.196 min     |
| HardSwish | 97.5183  | 97.6308| 30   | 6.269 min     |
| ReLU     | 97.2451  | 97.3375| 30   | 6.264 min     |

Positive activation percentage
| Run       | Smoothed | Value  | Step | Training Time |
|-----------|----------|--------|------|---------------|
| GELU      | 0.4191   | 0.4194 | 30   | 6.246 min     |
| LeakyReLU | 0.4276   | 0.4271 | 30   | 6.196 min     |
| HardSwish | 0.4034   | 0.404  | 30   | 6.269 min     |
| ReLU     | 0.4308   | 0.4316 | 30   | 6.264 min     |
### Combined Results

| Run         | Test Value | Train Value | Positive Activation Value |
|-------------|------------|-------------|---------------------------|
| GELU        | 98.6178    | 97.6358     | 0.4194                    |
| LeakyReLU   | 98.4876    | 97.4075     | 0.4271                    |
| HardSwish   | 98.728     | 97.6308     | 0.404                     |
| ReLU        | 98.4275    | 97.3375     | 0.4316                        |

LeakyReLU 0.125 LTrain:0.090833 ATrain: 97.20% LTest:0.051325 ATest: 98.44% Time[s]: 14.04 Act: 36.8%

LeakyReLU 0.06125 LTrain:0.086684 ATrain: 97.28% LTest:0.048970 ATest: 98.40% Time[s]: 14.35 Act: 38.7% w_

x * F.relu(x + 3) / 6 LTrain:0.075767 ATrain: 97.63% LTest:0.038514 ATest: 98.72% Time[s]: 12.85 Act: 43.2% 

x * F.relu(x + 1)  LTrain:0.082885 ATrain: 97.42% LTest:0.046766 ATest: 98.51% Time[s]: 16.99 Act: 53.1%

x * F.relu(x + 4)  LTrain:0.077152 ATrain: 97.57% LTest:0.043664 ATest: 98.67% Time[s]: 12.15 Act: 39.9% 

x * F.relu(x + 2) LTrain:0.076004 ATrain: 97.60% LTest:0.043627 ATest: 98.54% Time[s]: 12.64 Act: 49.0%

x * F.relu(x + 3) LTrain:0.074241 ATrain: 97.68% LTest:0.042816 ATest: 98.63% Time[s]: 12.86 Act: 44.2% 

x * F.relu(x + 0) LTrain:0.080196 ATrain: 97.50% LTest:0.049050 ATest: 98.54% Time[s]: 13.48 Act: 63.8%

F.relu(x + 0) - 0 LTrain:0.081163 ATrain: 97.46% LTest:0.048307 ATest: 98.50% Time[s]: 12.44 Act: 43.2%

F.relu(x + 1) - 1 LTrain:0.087294 ATrain: 97.30% LTest:0.049661 ATest: 98.44% Time[s]: 12.39 Act: 40.8% 

F.relu(x + 2) - 2 LTrain:0.115790 ATrain: 96.44% LTest:0.055633 ATest: 98.13% Time[s]: 12.72 Act: 44.2%

x * F.relu6(x + 3)  LTrain:0.075704 ATrain: 97.63% LTest:0.041766 ATest: 98.78% Time[s]: 12.41 Act: 41.5%  

x * F.relu6(x + 4) LTrain:0.079159 ATrain: 97.54% LTest:0.043536 ATest: 98.55% Time[s]: 12.41 Act: 38.8%

x * F.relu6(x + 2) LTrain:0.079153 ATrain: 97.49% LTest:0.044097 ATest: 98.55% Time[s]: 12.50 Act: 42.7% 

x * F.relu6(x + 1) LTrain:0.077949 ATrain: 97.54% LTest:0.051734 ATest: 98.39% Time[s]: 11.97 Act: 52.4%

x * F.relu6(x + 3) + NF4 LTrain:0.068223 ATrain: 97.87% LTest:0.038414 ATest: 98.72% Time[s]: 15.05 Act: 39.9%

F.relu(x) + NF4  LTrain:0.082073 ATrain: 97.45% LTest:0.047435 ATest: 98.50% Time[s]: 13.68 Act: 44.7% 

Can we swap to relu at inference time? Or is there also a significant inference time benefit? -> no we cant. switching activation drops accuracy, but healing is quick.

## Residual connections / Resnet

- Can a residual connection help to "scale down" noise, so that quantization noise is less significant?

- Helps to reduce training loss slightly for 4 bit, but only if scale value is learned as well. Same for 2 bit.

experiments on 64/64/64/20 epochs

```python	
    def forward(self, x):
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.ReLU(self.fc3(x))
        x = self.fc4(x)
        return x
```
*4bitsym*
Epoch [20/20], LTrain:0.070248 ATrain: 97.81% LTest:0.047278 ATest: 98.52% Time[s]: 13.93 Act: 39.5% w_clip/entropy[bits]: 0.474/2.63 0.412/3.39 0.330/3.60 0.657/3.64 

```python	
    def forward(self, x):
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(x+self.fc2(x))
        x = self.ReLU(x+self.fc3(x))
        x = self.fc4(x)
        return x
```
*4bitsym*
Epoch [20/20], LTrain:0.076385 ATrain: 97.59% LTest:0.040175 ATest: 98.72% Time[s]: 13.85 Act: 42.0% w_clip/entropy[bits]: 0.465/2.57 0.466/3.43 0.475/3.39 0.695/3.72 
*2bitsym*
Epoch [20/20], LTrain:0.110816 ATrain: 96.52% LTest:0.060259 ATest: 98.05% Time[s]: 14.13 Act: 41.0% w_clip/entropy[bits]: 0.379/1.51 0.376/1.85 0.293/1.91 0.597/1.93 


```python	
    def forward(self, x):
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = x + self.ReLU(self.fc2(x))
        x = x + self.ReLU(self.fc3(x))
        x = self.fc4(x)
        return x
```
Epoch [20/20], LTrain:0.079785 ATrain: 97.52% LTest:0.047587 ATest: 98.59% Time[s]: 13.16 Act: 33.0% w_clip/entropy[bits]: 0.451/2.64 0.502/3.41 0.492/3.51 0.791/3.76 

```python	
    def forward(self, x):
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = x * self.scale1 + self.ReLU(self.fc2(x))
        x = x * self.scale2 + self.ReLU(self.fc3(x))
        x = self.fc4(x)
        return x
```
*4bitsym*
Epoch [20/20], LTrain:0.068703 ATrain: 97.87% LTest:0.041398 ATest: 98.59% Time[s]: 13.85 Act: 31.9% w_clip/entropy[bits]: 0.505/2.59 0.410/3.46 0.415/3.40 0.651/3.74
*2bitsym*
Epoch [20/20], LTrain:0.107123 ATrain: 96.55% LTest:0.060191 ATest: 97.92% Time[s]: 13.86 Act: 34.0% w_clip/entropy[bits]: 0.384/1.51 0.383/1.86 0.341/1.83 0.565/1.98 

```python	
    def forward(self, x):
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x =  self.ReLU(x * self.scale1 + self.fc2(x))
        x =  self.ReLU(x * self.scale2 + self.fc3(x))
        x = self.fc4(x)
        return x
```
*4bitsym*
Epoch [20/20], LTrain:0.071519 ATrain: 97.79% LTest:0.042721 ATest: 98.65% Time[s]: 14.60 Act: 40.3% w_clip/entropy[bits]: 0.451/2.72 0.428/3.40 0.388/3.47 0.636/3.76 

*2bitsym*
Epoch [20/20], LTrain:0.106741 ATrain: 96.64% LTest:0.065592 ATest: 97.86% Time[s]: 14.45 Act: 43.3% w_clip/entropy[bits]: 0.375/1.53 0.376/1.85 0.346/1.85 0.562/1.97 

## Mixture-of-Experts

- Use routing mechanism to reduce the number of depth-wise conv2d to extract features

## Review clipping algorithm

- Re-Investigate whether OCTAV is really the best way to adjust clipping. The test accuracy seems to have degraded after introduction of OCTAV. 
- Potential approaches
  - Prioritize quantization noise over clipping noise as the network can
    adjust to clipping noise easier than to quantization noise. (Fewer weights involved)
  - Copilot suggestion (Why?): Use a fixed clipping value for the first layers and adjust the clipping value for the last layers. This would allow to use a higher clipping value for the first layers to extract more features and a lower clipping value for the last layers to reduce noise.
  - train clipping value as a parameter
  - decrease clipping value over time
  

## Refactor models.py

- See distillation branch for refactored models.py

## Implement Conv2d 

- Done with training code and exportquant.py
- TODO: Implement in C


## Investigate network capacity scaling also with data augmentation
- Is totnumofbits still limiting network capacity with augmentation? -> yes
- Does regularization with low bit quants help? -> a bit, (<5%)
- Once network is large enough, curves should saturate

## Implement pruning to optimize density

- Start with larger network than target, then use pruning/distillation

## Lottery ticket search



## 1.58 bit inference

- actually implement ternary inference

## 4.6 bit quantization

- https://www.mdpi.com/2227-7390/12/5/651
- https://github.com/cpldcpu/BitNetMCU/issues/4


# Parking lot

## Reorder tensors for better compressabiliy

- reorder layer tensors in a way where weights are more correlated while they are
  streamed in for MAC.

*Results*
- Reordering according to cosine similarity and L2 distance implemented.
- Does not seem to have sufficient effect on weight correlation
- Unforeseen issue: Delta encoding requires an additional bit, so entropy
is not reduced.


## Use modified ReLU to enforce sparse activations

Idea is to set to only pass the top 50% of positive activations.

```python	
class ReLUTopQ(nn.Module):
    def __init__(self):
        super(ReLUTopQ, self).__init__()
        
    def forward(self, x):
        # Handle batched inputs by taking max over all dimensions except batch
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        return 2 * F.relu(x - 0.5 * max_x)
```
Results:

Normal ReLU:
`Epoch [20/20], LTrain:0.072104 ATrain: 97.77% LTest:0.040317 ATest: 98.71% Time[s]: 19.01 Act: 39.4% w_clip/entropy[bits]: 0.468/2.65 0.387/3.49 0.333/3.58 0.562/3.80 `	
ReLUTopQ
`Epoch [20/20], LTrain:0.110103 ATrain: 96.56% LTest:0.061737 ATest: 98.15% Time[s]: 18.96 Act: 22.4% w_clip/entropy[bits]: 0.435/2.52 0.316/3.38 0.297/3.49 0.558/3.54 `

-> Activiation density is roughly reduced by factor of two, but loss is significantly higher. This does not work.

Pruning the bottom 50% of positive activations without rescaling does not work (no convergence).

## Optimize fc model

- mixed quantization, use 2 bit for first layers to compress data
- increase number of first layer features to extract
  - tested 2 bit first layer with 96 features and 4 bit for rest. No improvement.

## Stochastic weight averaging

- Average local minima for better generalization

  * Implemented in extra branch, performance quite mixed so far.
  * One issue is that there is no straightforward way to average weights in combination with quantization. Averaging will move away the weights from the quantization levels, which will add noise. 
  * So far, SWA is usually worse than the normally trained model.
  * Astonishingly, the cyclic learning rate does not seem to be significantly worse than the cosine schedule on standard model.
```
SWA model updated
Epoch [85/90], Tr_loss:0.060427 Tr_acc: 98.07% Te_loss:0.041853 Te_acc: 98.62% SWA_loss:0.039543 SWA_acc: 98.75% Time[s]: 19.46 w_clip/entropy[bits]: 1.428/3.59 1.168/3.86 0.931/3.73 1.057/3.88 
Epoch [86/90], Tr_loss:0.110980 Tr_acc: 96.49% Te_loss:0.060551 Te_acc: 98.12% SWA_loss:0.039543 SWA_acc: 98.75% Time[s]: 19.45 w_clip/entropy[bits]: 1.428/3.63 1.168/3.85 0.931/3.70 1.057/3.87 
Epoch [87/90], Tr_loss:0.094380 Tr_acc: 96.97% Te_loss:0.056176 Te_acc: 98.30% SWA_loss:0.039543 SWA_acc: 98.75% Time[s]: 19.60 w_clip/entropy[bits]: 1.428/3.64 1.168/3.85 0.931/3.69 1.057/3.88 
Epoch [88/90], Tr_loss:0.083052 Tr_acc: 97.37% Te_loss:0.051770 Te_acc: 98.47% SWA_loss:0.039543 SWA_acc: 98.75% Time[s]: 19.35 w_clip/entropy[bits]: 1.428/3.64 1.168/3.84 0.931/3.70 1.057/3.88 
Epoch [89/90], Tr_loss:0.069970 Tr_acc: 97.72% Te_loss:0.043877 Te_acc: 98.56% SWA_loss:0.039543 SWA_acc: 98.75% Time[s]: 19.48 w_clip/entropy[bits]: 1.428/3.63 1.168/3.85 0.931/3.70 1.057/3.87 
SWA model updated
Epoch [90/90], Tr_loss:0.060022 Tr_acc: 98.09% Te_loss:0.038376 Te_acc: 98.83% SWA_loss:0.038772 SWA_acc: 98.74% Time[s]: 19.35 w_clip/entropy[bits]: 1.428/3.63 1.168/3.85 0.931/3.71 1.057/3.87 
```

## Regularization by switching quantization levels

- starting with both higher and lower quantization tha the target seems to cause higher loss.                                                                                                                                                                  
- Tried: start with 2bit, switch to 4bit at ep30
- start with 8 bit switch to 4 bit at ep10 and ep30 -> lower loss when switching at ep10

**code**

```python	

        if epoch == 10:
            switch_quantization(model, 'NF4')

def switch_quantization(model, new_QuantType):
    for module in model.modules():
        if isinstance(module, (BitLinear, BitConv2d)):
            module.QuantType = new_QuantType
          
```

## Self-compressing models

https://github.com/geohot/ai-notebooks/blob/master/mnist_self_compression.ipynb
https://twitter.com/realGeorgeHotz/status/1819963680739512550
https://arxiv.org/pdf/2301.13142
https://www.cs.toronto.edu/~hinton/absps/colt93.pdf
https://openreview.net/forum?id=HkgxW0EYDS
https://arxiv.org/abs/1810.00440

- Some simpler ideas:
  - Compress feature detection (unlearn redundant feature detectors to reduce computational effort in depthwise CNN)
- Per layer auto-quantization
  - auto-scaling not needed

### Results

- Implemented as per-layer quantization learning. One learnable parameter per layer for bit precision, but no parameter for scale due to normalization.
  - Result: Model tends to overweight this parameter due to strong influence on loss function, "starves itself" to lower bit precision.
- Quantization learning probably works better if implemented per channel, but this is not easily implemented in inference code.
- Without quantization learning, the self-compression methods become equal to pruning -> investigate pruning

```python
        elif self.QuantType == 'SelfCompress':
            u = (w * scale).round().clamp( -(2.0**(self.bpw-1)) , 2.0**(self.bpw-1) - 1)     
...
        elif self.QuantType == 'SelfCompress':
            self.bpw = nn.Parameter(torch.tensor(8.0))  # Start with 16 bit precision
...
def calculate_compression_loss(model):
    total_bits = 0
    total_weights = 0
    for module in model.modules():
        if isinstance(module, (BitLinear, BitConv2d)):
            if module.QuantType == 'SelfCompress':
                total_bits += module.bpw.mean() * module.weight.numel()
                total_weights += module.weight.numel()
    return total_bits / total_weights
```

## Distilliation

- Idea: Distill CNN with better feature detection to fc model. Last layer (classification layer) is frozen and student model is trained with distillation loss from feature detection.
- See distillation branch 
- Training works, but no benefit observed. Generally seems a bit instable, may require some bug fixing.

# Done

## Implement OCTAV
- Nvidias approach to achieve good FP4 accuracy in the B200?  **Done**
- Add documentation **Done**

## Refactor code to introduce clipping parameter
- Freeze clipping parameter at some point? Introduce as option. -> **done**

## Implement two step learning schedule
- Use cosine schedule, but half LR after 50% of epochs
- Used in Bitnet paper and matmulless paper

   1) LR=0.003, half at ep30
   2) LR=0.001, no halving
   
-> **done**, slighty improved regularisation

```python
    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
``` 

## Benchmark Layernorm vs RMSnorm

- Layernorm seems to be better than RMSnorm for very small networks where it improves convergence.
  - using full layernorm is slightly worse than RMSnorm
    - Argument: Removing bias removes information?
  - To fix convergence in small networks, it is sufficient to use layernorm before first layer
  - Layernorm increases nonzero activations from ~40% to ~50%
-> stick with RMSnorm for now
