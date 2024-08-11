import torch
import torch.nn as nn
import torch.nn.functional as F
from BitNetMCU import BitLinear, BitConv2d

   

class MaskingLayer(nn.Module):

    def __init__(self, num_channels):
        super(MaskingLayer, self).__init__()
        # self.mask = nn.Parameter(torch.zeros(num_channels))  # Initialize to zeros
        self.mask = nn.Parameter(torch.ones(num_channels))  # Initialize to zeros

    def forward(self, x):
        # mask = self.sigmoid(self.mask)  # Apply sigmoid to get values between 0 and 1
        mask = self.mask
        return x * mask.view(1, -1)
   
    def prune_channels(self, prune_number=8, groups=0):
        with torch.no_grad():
            if groups > 0:
                channels_per_group = self.mask.size(0) // groups
                prune_per_group = prune_number // groups

                mask_values = self.mask.clone()
                for group in range(groups):
                    start = group * channels_per_group
                    end = start + channels_per_group
                    group_mask_values, _ = torch.sort(self.mask[start:end].view(-1))
                    threshold = group_mask_values[prune_per_group - 1].item()
                    mask_values[start:end] = (self.mask[start:end] > threshold).float()
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

        # self.flatten = nn.Flatten()

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

            # 128ch out , 99.3%
            # BitConv2d(1, 32, kernel_size=3, stride=1, padding=(0,0),  groups=1,QuantType='8bit',NormType='None', WScale=WScale),
            # nn.ReLU(),
            # BitConv2d(32, 32, kernel_size=3, stride=1, padding=(0,0),  groups=32,QuantType='8bit',NormType='None', WScale=WScale),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),           
            # BitConv2d(32, 32, kernel_size=3, stride=1, padding=(0,0),  groups=32,QuantType='8bit',NormType='None', WScale=WScale),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),           


            # 256ch out , 99.5%
            BitConv2d(1, 64, kernel_size=3, stride=1, padding=(0,0),  groups=1,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            BitConv2d(64, 64, kernel_size=3, stride=1, padding=(0,0),  groups=64,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           
            BitConv2d(64, 64, kernel_size=3, stride=1, padding=(0,0),  groups=64,QuantType='8bit',NormType='None', WScale=WScale),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           


            # 256ch out, 99.4%
            # BitConv2d(1, 16, kernel_size=3, stride=1, padding=(0,0),  groups=1,QuantType='8bit',NormType='None', WScale=WScale),
            # nn.ReLU(),
            # BitConv2d(16, 16, kernel_size=3, stride=1, padding=(0,0),  groups=16,QuantType='8bit',NormType='None', WScale=WScale),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2),           

            # BitConv2d(16, 16, kernel_size=3, stride=1, padding=(0,0),  groups=16,QuantType='8bit',NormType='None', WScale=WScale),
            # nn.ReLU(),

            nn.Flatten(),
            # MaskingLayer(256),   # learnable masking layer for auto-pruning
            BitLinear(256 , network_width1,QuantType='2bitsym',NormType=NormType, WScale=WScale),
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

    
class MAXMNIST(nn.Module):
    """
    CNN+FC Neural Network for MNIST dataset with full conv2d

    """
    def __init__(self,network_width1=64,network_width2=64,network_width3=64,QuantType='Binary',WScale='PerTensor',NormType='RMS'):
        super(MAXMNIST, self).__init__()

        self.network_width1 = network_width1
        self.network_width2 = network_width2
        self.network_width3 = network_width3

        # Important!!! The layers will be processed by the quantized class in the order they are defined in the __init__ function
        # So the first layer should be the first layer in the network, and so on.

        self.conv1 = BitConv2d(1, 16, kernel_size=3, stride=1, padding=(0,0),  groups=1,QuantType='8bit',NormType='None', WScale=WScale)
        self.conv1b = BitConv2d(16, 16, kernel_size=3, stride=1, padding=(0,0),  groups=1,QuantType='8bit',NormType='None', WScale=WScale)
        self.conv2 = BitConv2d(16, 96, kernel_size=12, stride=1, padding=(0,0), groups=16,QuantType=QuantType,NormType='None', WScale=WScale)
     
        self.flatten = nn.Flatten()
    
        self.fc1 = BitLinear(96 , network_width1,QuantType=QuantType,NormType=NormType, WScale=WScale)
        self.fc2 = BitLinear(network_width1, network_width2,QuantType=QuantType,NormType=NormType, WScale=WScale)

        if network_width3>0:
            self.fc3 = BitLinear(network_width2, network_width3,QuantType=QuantType,NormType=NormType, WScale=WScale)
            self.fcl = BitLinear(network_width3, 10,QuantType=QuantType,NormType=NormType, WScale=WScale)
        else:
            self.fcl = BitLinear(network_width2, 10,QuantType=QuantType,NormType=NormType, WScale=WScale)

        # self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        # x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv1b(x))

        x = F.relu(self.conv2(x))        

        x = self.flatten(x)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.network_width3>0:
            x = F.relu(self.fc3(x))

        # x = self.dropout(x)
        x = self.fcl(x)
        return x

