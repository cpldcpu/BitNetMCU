
# Funnel

 
## Implement Conv2d 

- Done with training code and exportquant.py
- TODO: Implement in C

## Implement two step learning schedule
- Use cosine schedule, but half LR after 50% of epochs
- Used in Bitnet paper and matmulless paper

## Investigate network capacity scaling also with data augmentation
- Is totnumofbits still limiting network capacity with augmentation? -> yes
- Does regularization with low bit quants help? -> a bit, (<5%)
- Once network is large enough, curves should saturate

## Implement pruning to optimize density

- Start with larger network than target, then use pruning/distillation

## Lottery ticket search

## Stochastic weight averaging

- Average local minima for better generalization

## 1.58 bit inference

- actually implement ternary inference

## 4.6 bit quantization

- https://www.mdpi.com/2227-7390/12/5/651
- https://github.com/cpldcpu/BitNetMCU/issues/4
- 

# Parking lot

## Optimize fc model

- mixed quantization, use 2 bit for first layers to compress data
- increase number of first layer features to extract
  - tested 2 bit first layer with 96 features and 4 bit for rest. No improvement.

# Done

## Implement OCTAV
- Nvidias approach to achieve good FP4 accuracy in the B200?  **Done**
- Add documentation **Done**

## Refactor code to introduce clipping parameter
- Freeze clipping parameter at some point? Introduce as option. -> **done**

