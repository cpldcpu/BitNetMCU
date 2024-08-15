import torch
import torch.nn as nn
import torch.nn.functional as F
from BitNetMCU import BitLinear, BitConv2d

class MaskingLayer(nn.Module):

    def __init__(self, num_channels):
        super(MaskingLayer, self).__init__()
        self.mask = nn.Parameter(torch.ones(num_channels))  

    def forward(self, x):
        return x * self.mask.view(1, -1)´6ß
   
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
    def __init__(self,network_width1=64,network_width2=64,network_width3=64,QuantType='Binary',WScale='PerTensor',NormType='RMS'):
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

        self.classifier= BitLinear(network_width2, 10,QuantType=QuantType,NormType=NormType, WScale=WScale)

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
    def __init__(self,network_width1=64,network_width2=64,network_width3=64,QuantType='Binary',WScale='PerTensor',NormType='RMS'):
        super(CNNMNIST, self).__init__()

        self.network_width1 = network_width1
        self.network_width2 = network_width2
        self.network_width3 = network_width3

        self.model = nn.Sequential(

            # 256ch out , 99.5%
            BitConv2d(1, 64, kernel_size=3, stride=1, padding=(0,0),  groups=1,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            BitConv2d(64, 64, kernel_size=3, stride=1, padding=(0,0),  groups=64,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           
            BitConv2d(64, 64, kernel_size=3, stride=1, padding=(0,0),  groups=64,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           

            nn.Flatten(),
            # MaskingLayer(256+128),   # learnable masking layer for auto-pruning
            BitLinear(256 , network_width1,QuantType='4bitsym',NormType=NormType, WScale=WScale),
            nn.ReLU(),
            BitLinear(network_width1, network_width2,QuantType=QuantType,NormType=NormType, WScale=WScale),
            nn.ReLU()
        )

        if network_width3>0:
            self.model.add_module("fc3", BitLinear(network_width2, network_width3,QuantType=QuantType,NormType=NormType, WScale=WScale))
            self.model.add_module("relu_fc2", nn.ReLU())

        self.classifier= BitLinear(network_width2, 10,QuantType=QuantType,NormType=NormType, WScale=WScale)
        # self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
