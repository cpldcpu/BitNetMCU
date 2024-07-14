
# Refactor code to introduce clipping parameter
- Freeze clipping parameter at some point? Introduce as option.

# Implement OCTAV
- Nvidias approach to achieve good FP4 accuracy in the B200?

# Implement two step learning schedule
- Use cosine schedule, but half LR after 50% of epochs
- Used in Bitnet paper and matmulless paper

# Investigate network capacity scaling also with data augmentation
- Is totnumofbits still limiting network capacity with augmentation?
- Does regularization with low bit quants help?
- Once network is large enough, curves should saturate

# Implement pruning to imptimize densitiy
- Start with larger network then target, then use pruning/distillation

# Lottery ticket search


# Stochastic weight averaging

- Average local minima for better generalization