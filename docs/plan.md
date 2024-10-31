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

## Benchmark Layernorm vs RMSnorm

- Layernorm seems to be better than RMSnorm for very small networks where it improves convergence.
  - Unclear whether it helps for networks that are at capacity.
  - Does it help with generalization? 
  - What is the penalty in terms of computational effort as there are fewer (no) zero activations?

## Mixture-of-Experts

- Use routing mechanism to reduce the number of depth-wise conv2d to extract features


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


