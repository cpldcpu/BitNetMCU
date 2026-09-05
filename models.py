import torch
import torch.nn as nn
import torch.nn.functional as F
from BitNetMCU import BitLinear, BitConv2d

class MaskingLayer(nn.Module):

    def __init__(self, num_channels):
        super(MaskingLayer, self).__init__()
        self.mask = nn.Parameter(torch.ones(num_channels))  

    def forward(self, x):
        return x * self.mask.view(1, -1)
   
    def prune_channels(self, prune_number=8, groups=0):
        with torch.no_grad():
            if groups > 0:

                channels_per_group = self.mask.size(0) // groups
                group_mask_values = torch.zeros(groups)

                # Calculate the sum of mask values for each group
                for group in range(groups):
                    start = group * channels_per_group
                    end = start + channels_per_group
                    group_mask_values[group] = self.mask[start:end].sum()

                # Sort the group mask values and determine the threshold
                sorted_group_mask_values, _ = torch.sort(group_mask_values)
                threshold = sorted_group_mask_values[prune_number - 1].item()

                # Update the mask values to prune entire groups
                mask_values = self.mask.clone()
                for group in range(groups):
                    start = group * channels_per_group
                    end = start + channels_per_group
                    if group_mask_values[group] <= threshold:
                        mask_values[start:end] = 0.0
                    else:
                        mask_values[start:end] = 1.0                
            else:
                sorted_mask_values, _ = torch.sort(self.mask.view(-1))
                threshold = sorted_mask_values[prune_number - 1].item()
                mask_values = (self.mask > threshold).float()

            self.mask.requires_grad = False
            self.mask.data = mask_values

        pruned_channels = (mask_values < threshold).sum().item()
        remaining_channels = (mask_values >= threshold).sum().item()
        print(f"Pruned {pruned_channels} channels. {remaining_channels} channels remaining.")
        return pruned_channels, remaining_channels

          
class FCMNIST(nn.Module):
    """
    Fully Connected Neural Network for MNIST dataset.
    16x16 input image, 3 hidden layers with a configurable width.

    @cpldcpu 2024-March-24

    """
    def __init__(self,network_width1=64,network_width2=64,network_width3=64,QuantType='Binary',WScale='PerTensor',NormType='RMS', num_classes: int = 10):
        super(FCMNIST, self).__init__()

        self.network_width1 = network_width1
        self.network_width2 = network_width2
        self.network_width3 = network_width3

        self.model = nn.Sequential(
            nn.Flatten(),
            BitLinear(1* 16 *16, network_width1,QuantType=QuantType,NormType=NormType, WScale=WScale),
            nn.ReLU(),
            BitLinear(network_width1, network_width2,QuantType=QuantType,NormType=NormType, WScale=WScale),
            nn.ReLU()
        )

        if network_width3>0:
            self.model.add_module("fc3", BitLinear(network_width2, network_width3,QuantType=QuantType,NormType=NormType, WScale=WScale))
            self.model.add_module("relu_fc2", nn.ReLU())

        last_width = network_width3 if network_width3>0 else network_width2
        # Output layer parameterized by number of classes (default 10 for MNIST / 47 for EMNIST balanced, etc.)
        self.classifier= BitLinear(last_width, num_classes,QuantType=QuantType,NormType=NormType, WScale=WScale)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)

        return x
    
class CNNMNIST(nn.Module):
    """
    CNN+FC Neural Network for MNIST dataset. Depthwise separable convolutions.
    16x16 input image, 3 hidden layers with a configurable width.

    @cpldcpu 2024-April-19

    """
    def __init__(self,network_width1=64,network_width2=64,network_width3=64,cnn_width=64,QuantType='Binary',WScale='PerTensor',NormType='RMS', num_classes: int = 10):
        super(CNNMNIST, self).__init__()

        self.network_width1 = network_width1
        self.network_width2 = network_width2
        self.network_width3 = network_width3
        self.cnn_width = cnn_width

        self.model = nn.Sequential(

            # 256ch out , 99.5%
            BitConv2d(1, cnn_width, kernel_size=3, stride=1, padding=(0,0),  groups=1,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            BitConv2d(cnn_width, cnn_width, kernel_size=3, stride=1, padding=(0,0),  groups=cnn_width,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BitConv2d(cnn_width, cnn_width, kernel_size=3, stride=1, padding=(0,0),  groups=cnn_width,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            # MaskingLayer(96*4),   # learnable masking layer for auto-pruning
            BitLinear(cnn_width*4 , network_width1,QuantType='2bitsym',NormType=NormType, WScale=WScale),
            nn.ReLU(),
            BitLinear(network_width1, network_width2,QuantType=QuantType,NormType=NormType, WScale=WScale),
            nn.ReLU()
        )

        if network_width3>0:
            self.model.add_module("fc3", BitLinear(network_width2, network_width3,QuantType=QuantType,NormType=NormType, WScale=WScale))
            self.model.add_module("relu_fc2", nn.ReLU())

        last_width = network_width3 if network_width3>0 else network_width2
        # Output layer parameterized by number of classes (default 10 for MNIST / 47 for EMNIST balanced, etc.)
        self.classifier= BitLinear(last_width, num_classes,QuantType=QuantType,NormType=NormType, WScale=WScale)
        # self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x




class PatchMNIST(nn.Module):
    """
    Patch front-end + dense head (paper "Everything is a Table", sec. 5.6 / Table 3, extended).

    The 16x16 input is cut into patch_size x patch_size windows (patch_mode='block'; patch_stride < patch_size
    gives overlapping windows, e.g. 8x8 at stride 4 -> 3x3 = 9 positions) or polyphase views
    (patch_mode='polyphase', the control that shares weights without locality). Every window is flattened to a
    token and passed through a stem of 1-3 BitLinear layers (patch_width1 [-> patch_width2 [-> patch_width3]],
    ReLU between); patch_shared=True uses ONE stem for all windows (a large-kernel convolution expressed with the
    fc kernel), False gives every window its own stem. The window outputs are concatenated (nblocks * last stem
    width) and fed to a dense head network_width1 -> network_width2 [-> network_width3] -> num_classes.

    patch_shared_scale=True shares one activation scale/shift per image across the window tokens (act_group) -
    required for MCU-exact training, where it preserves the relative magnitudes of the windows.

    On the MCU the shared stem is the ordinary fc kernel called once per window with the same weight pointer;
    per-window quantization falls out of the per-token machinery (each window is a token).

    Reference points (NF4/A4, MCU-exact, 60 epochs): shared 8x8 quadrants 64-32-32, head 96: 25.5k weights, 99.16%;
    overlap stride 4, 64-32-16, head 64: 16.5k, 99.25%; overlap stride 2, 64-32-8, head 64: 19.8k, 99.37%.
    """
    def __init__(self, network_width1=96, network_width2=96, network_width3=0, patch_width1=32, patch_width2=0, patch_width3=0,
                 patch_mode='block', patch_shared_scale=False, patch_shared=True, patch_size=8, patch_stride=0, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', num_classes: int = 10):
        super().__init__()
        assert patch_mode in ('block', 'polyphase')
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride > 0 else patch_size   # stride < size: overlapping windows (block mode only)
        if self.patch_stride == patch_size:
            assert 16 % patch_size == 0
            self.grid = 16 // patch_size                   # blocks per side
        else:
            assert patch_mode == 'block' and (16 - patch_size) % self.patch_stride == 0
            self.grid = (16 - patch_size) // self.patch_stride + 1   # e.g. 8x8 windows at stride 4 -> 3x3 positions
        self.nblocks = self.grid ** 2                      # 4 for 8x8 patches, 16 for 4x4, 9 for 8x8/stride 4
        self.patch_width3 = patch_width3
        self.patch_shared_scale = patch_shared_scale
        self.patch_shared = patch_shared      # False: every block gets its own stem weights (4x the stem parameters)
        self.patch_mode = patch_mode
        self.patch_width1, self.patch_width2 = patch_width1, patch_width2
        q = dict(QuantType=QuantType, NormType=NormType, WScale=WScale)
        nb, pin = self.nblocks, patch_size * patch_size
        if patch_shared:
            self.stem1 = BitLinear(pin, patch_width1, **q)
            self.stem2 = BitLinear(patch_width1, patch_width2, **q) if patch_width2 > 0 else None
            self.stem3 = BitLinear(patch_width2, patch_width3, **q) if patch_width3 > 0 else None
        else:
            self.stem1 = nn.ModuleList([BitLinear(pin, patch_width1, **q) for _ in range(nb)])
            self.stem2 = nn.ModuleList([BitLinear(patch_width1, patch_width2, **q) for _ in range(nb)]) if patch_width2 > 0 else None
            self.stem3 = nn.ModuleList([BitLinear(patch_width2, patch_width3, **q) for _ in range(nb)]) if patch_width3 > 0 else None
        if patch_shared_scale:
            assert patch_shared, "patch_shared_scale needs shared stems (grouped tokens)"
            # one activation scale / shift per image for the block tokens (see BitQuant.act_group)
            for st in (self.stem1, self.stem2, self.stem3):
                if st is not None:
                    st.act_group = nb
        stem_out = nb * (patch_width3 if patch_width3 > 0 else patch_width2 if patch_width2 > 0 else patch_width1)
        self.model = nn.Sequential(
            BitLinear(stem_out, network_width1, **q), nn.ReLU(),
            BitLinear(network_width1, network_width2, **q), nn.ReLU())
        if network_width3 > 0:
            self.model.add_module("fc3", BitLinear(network_width2, network_width3, **q))
            self.model.add_module("relu_fc3", nn.ReLU())
        last = network_width3 if network_width3 > 0 else network_width2
        self.classifier = BitLinear(last, num_classes, **q)

    def patches(self, x):
        """x: [N,1,16,16] -> [N*nblocks, patch_size^2] tokens, row-major window order (block-major within an image)."""
        n, g, ps = x.shape[0], self.grid, self.patch_size
        if self.patch_mode == 'block' and self.patch_stride != ps:
            # overlapping windows: unfold -> [N, ps*ps, g*g] -> [N*g*g, ps*ps] (row-major positions)
            return F.unfold(x, kernel_size=ps, stride=self.patch_stride).transpose(1, 2).reshape(n * g * g, ps * ps)
        if self.patch_mode == 'block':
            p = x.reshape(n, 1, g, ps, g, ps).permute(0, 2, 4, 1, 3, 5)      # [N, g, g, 1, ps, ps]
        else:
            p = x.reshape(n, 1, ps, g, ps, g).permute(0, 3, 5, 1, 2, 4)      # [N, g, g, 1, ps, ps] stride-g views
        return p.reshape(n * g * g, ps * ps)

    def forward(self, x):
        n = x.shape[0]
        if self.patch_shared:
            t = F.relu(self.stem1(self.patches(x)))
            if self.stem2 is not None:
                t = F.relu(self.stem2(t))
            if self.stem3 is not None:
                t = F.relu(self.stem3(t))
            t = t.reshape(n, -1)
        else:
            blocks = self.patches(x).reshape(n, self.nblocks, -1)
            outs = []
            for b in range(self.nblocks):
                tb = F.relu(self.stem1[b](blocks[:, b]))
                if self.stem2 is not None:
                    tb = F.relu(self.stem2[b](tb))
                if self.stem3 is not None:
                    tb = F.relu(self.stem3[b](tb))
                outs.append(tb)
            t = torch.cat(outs, dim=1)
        return self.classifier(self.model(t))
