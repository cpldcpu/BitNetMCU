import torch.nn as nn
import torch.nn.functional as F
from BitNetMCU import BitLinear

class FCMNIST(nn.Module):
    """
    Fully Connected Neural Network for MNIST dataset.
    16x16 input image, 3 hidden layers with a configurable width.

    @cpldcpu 2024-March-24

    """
    def __init__(self,network_width1=64,network_width2=64,network_width3=64,QuantType='Binary',WScale='PerTensor',NormType='RMS', quantscale=0.25):
        super(FCMNIST, self).__init__()

        self.network_width1 = network_width1
        self.network_width2 = network_width2
        self.network_width3 = network_width3
        self.quantscale = quantscale

        self.fc1 = BitLinear(1* 1 *16 *16, network_width1,QuantType=QuantType,NormType=NormType, WScale=WScale, quantscale=quantscale)
        self.fc2 = BitLinear(network_width1, network_width2,QuantType=QuantType,NormType=NormType, WScale=WScale , quantscale=quantscale )
        if network_width3>0:
            self.fc3 = BitLinear(network_width2, network_width3,QuantType=QuantType,NormType=NormType, WScale=WScale , quantscale=quantscale)
            self.fcl = BitLinear(network_width3, 10,QuantType=QuantType,NormType=NormType, WScale=WScale , quantscale=quantscale)
        else:
            self.fcl = BitLinear(network_width2, 10,QuantType=QuantType,NormType=NormType, WScale=WScale , quantscale=quantscale)
            
        # self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.network_width3>0:
            x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fcl(x)
        return x
    
