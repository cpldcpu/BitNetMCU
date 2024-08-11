import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# @cpldcpu 2024-June-2
    
class BitQuant:
    """
    Class to handle quantization of activations and weights.

    Quantization Types:
    - Binary         : 1 bit
    - Ternary        : 1.58 bits
    - BinaryBalanced : 1 bit, weights are balanced around zero
    - 2bitsym        : 2 bit symmetric
    - 4bitsym        : 4 bit symmetric
    - FP130          : 4 bit shift encoding
    - NF4            : 4 bit non-linear quantization (NormalFloat4)
    - 8bit           : 8 bit

    WScale 
    - PerTensor      : The weight scaling is calculated per Tensor
    - PerOutput      : The weight scaling is calculated per Output

    Implementation was initially based on:
    https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

    """
    def __init__(self, QuantType='Binary', WScale='PerTensor'):
        self.QuantType = QuantType
        self.WScale = WScale
        self.s = torch.nn.Parameter(torch.tensor(1.0))
        self.s.requires_grad = False  # no gradient for clipping scalar

        if self.QuantType in ['Binary', 'BinarySym']:
            self.bpw = 1
        elif self.QuantType in ['2bitsym']: 
            self.bpw = 2
        elif self.QuantType in ['Ternary']: 
            self.bpw = 1.6
        elif self.QuantType in ['4bit', '4bitsym', 'FP130' , 'NF4']:
            self.bpw = 4
        elif self.QuantType == '5bitsym':
            self.bpw = 5
        elif self.QuantType == '8bit':
            self.bpw = 8
        else:
            raise AssertionError(f"Invalid QuantType: {self.QuantType}")

        if not self.WScale in ['PerOutput', 'PerTensor']:
            raise AssertionError(f"Invalid WScale: {self.WScale}. Expected one of: 'PerTensor', 'PerOutput'")
    
    # Octave optimum clipping algorithm (C. Sakr et al., 2022)
    # see https://arxiv.org/abs/2206.06501
    def octav(self, tensor, num_iterations=10, s=-1):
        if s<0:
            # s = torch.sum(torch.abs(tensor)) / torch.sum(tensor != 0)
            s = tensor.abs().mean().clamp_(min=1e-5) * 0.25 # blind estimate as starting point
        for _ in range(num_iterations):
            indicator_le = (torch.abs(tensor) <= s).float()
            indicator_gt = (torch.abs(tensor) > s).float()
            numerator = torch.sum(torch.abs(tensor) * indicator_gt)
            denominator = (4**-self.bpw / 3) * torch.sum(indicator_le) + torch.sum(indicator_gt)
            s = numerator / denominator
        return s 

    def update_clipping_scalar(self, w, algorithm='octav', quantscale=0.25):
        """ 
        Update the weight scale factor for the quantization.
        Args:
            w:          a weight tensor with shape [d, k]
            algorithm:  clipping algorithm to use
                'octav' : Octave optimum clipping algorithm
                'prop'  : Proportional clipping algorithm
        Returns:
            s:          updated clipping scalar for the quantization
        """
    
        s= self.s

        if algorithm == 'octav':
            if self.WScale=='PerOutput':
                s = torch.stack([self.octav(row, 10) for row in w])
            else:
                s = self.octav(w, 10, s)             
        elif algorithm == 'prop':
            if self.WScale=='PerOutput':
                s = w.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5) / quantscale 
            else:
                s = w.abs().mean().clamp_(min=1e-5) / quantscale 
        else:
            raise AssertionError(f"Invalid algorithm: {algorithm}. Expected one of: 'octav', 'prop'")

        self.s = torch.nn.Parameter(s)
        self.s.requires_grad = False  # no gradient for clipping scalar

        return s

    def activation_quant(self, x):
        """ Per-token quantization to 8 bits. No grouping is needed for quantization.
        Args:
        x: an activation tensor with shape [n, d]
        Returns:
        y: a quantized activation tensor with shape [n, d]
        scale: scale factor for the quantization
        """
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) 
        return y, scale

    def weight_quant(self, w):
        """ Per-tensor quantization.
        Args:
        w: a weight tensor with shape [d, k]
        Returns:
        u:     a quantized weight with shape [d, k]
        scale: scale factor for the quantization
        bpw:   bit per weight
        """

        if self.QuantType == 'FP130':
           scale = 128.0 / self.s
        elif self.QuantType == 'NF4':
            scale = 1.0 / self.s
        elif self.QuantType == 'Ternary': # 1.58bits
            # scale = 1.0 / self.s
            scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        else:
            scale = (2.0**(self.bpw-1)) / self.s

        if self.QuantType == 'Ternary': # 1.58bits
            u = (w * scale ).round().clamp_(-1, 1) 
        elif self.QuantType == 'Binary': # 1 bit
            e = w.mean()
            u = (w - e).sign() 
        elif self.QuantType == 'BinarySym': # 1 bit
            u = w.sign() 
        elif self.QuantType == '2bitsym':
            u = ((w * scale - 0.5).round().clamp_(-2, 1) + 0.5) 
        elif self.QuantType == '4bit': # 4 bit in one-complement encoding for inference with multiplication
            # u = (w * scale).round().clamp_(-8, 7) # no convergence with this?!
            u = ((w * scale - 0.01).round().clamp_(-8, 7) + 0.01)  
        elif self.QuantType == '4bitsym':
            u = ((w * scale - 0.5).round().clamp_(-8, 7) + 0.5)  
        elif self.QuantType ==  'FP130': # encoding (F1.3.0) : S * ( 2^E3 + 1) -> min 2^0 = 1, max 2^7 = 128
            e = ((w * scale).abs()).log2().floor().clamp_(0, 7)
            u = w.sign()*(e.exp2())    
        elif self.QuantType == 'NF4':
            # NF4 levels (16 levels for 4 bits)
            levels = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 
                                   0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.723, 1.0], device=w.device)
            u , _ = self.quantize_list(w * scale, levels)
        elif self.QuantType == '5bitsym':
            u = ((w * scale - 0.5).round().clamp_(-16, 15) + 0.5)
        elif self.QuantType == '8bit': # -128 to 127
            u = (w * scale).round().clamp_(-128, 127) 
        else:
            raise AssertionError(f"Invalid QuantType: {self.QuantType}. Expected one of: 'Binary', 'BinaryBalanced', '2bitsym', '4bitsym', '8bit'")
        
        return u, scale, self.bpw

    def quantize_list(self, x, levels):
        """
        Quantize the input tensor x to the nearest level in the levels list.
        """        
        # Compute the absolute difference between x and each level
        diff = torch.abs(x.unsqueeze(-1) - levels)
        # Find the index of the closest level for each element in x
        indices = torch.argmin(diff, dim=-1)

        return levels[indices], indices
    
class BitLinear(nn.Linear, BitQuant):
    """
    Linear fully connected layer with quantization aware training and normalization.
    Configurable quantization and normalization types.

    Normalization Types:
    - RMS            : Root Mean Square
    - Lin            : L1 Norm
    - BatchNorm      : Batch Normalization
    
    This is not optimized for speed or efficiency...

    @cpldcpu 2024-March-24
    """
    def __init__(self, in_features, out_features, bias=False, QuantType='Binary', WScale='PerTensor', NormType='RMS'):
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        BitQuant.__init__(self, QuantType, WScale)

        self.NormType = NormType

    def forward(self, x):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, k]
        """
        w = self.weight # a weight tensor with shape [d, k]
        x_norm = self.Normalize(x)

        if self.QuantType == 'None':
            y = F.linear(x_norm, w)
        else:
            # A trick for implementing Straight-Through-Estimator (STE) using detach()
            x_int, x_scale = self.activation_quant(x_norm)
            x_quant = x_norm + ( x_int / x_scale - x_norm).detach()

            w_int, w_scale, _ = self.weight_quant(w)
            w_quant = w + (w_int / w_scale - w).detach()

            y = F.linear(x_quant, w_quant)
        return y
    
    def Normalize(self, x):
        """ Normalization. Normalizes along the last dimension -> different normalization value for each activation vector.
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: a normalized tensor with shape [n, d]
        """
        if self.NormType == 'RMS':
            y = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            z =x / y
        elif self.NormType == 'Lin':
            y = torch.mean(torch.abs(x), dim=-1, keepdim=True)
            z =x / y
        elif self.NormType == 'BatchNorm':
            z = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-5)
        else:
            raise AssertionError(f"Invalid NormType: {self.NormType}. Expected one of: 'RMS', 'Lin', 'BatchNorm'")
        return z

class BitConv2d(nn.Conv2d, BitQuant):
    """
    2D convolution layer with quantization aware training and normalization.
    Configurable quantization and normalization types.

    Normalization Types:
    - RMS            : Root Mean Square
    - None           : No normalization

    @cpldcpu 2024-June-2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1,  QuantType='4bitsym', WScale='PerTensor', NormType='RMS'):
        nn.Conv2d.__init__(self,in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        BitQuant.__init__(self, QuantType, WScale)

        self.NormType = NormType
        self.groups = groups
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, k]
        """
        w = self.weight # a weight tensor with shape [d, k]
        x_norm = self.Normalize(x)

        if self.QuantType == 'None':
            y = F.conv2d(x_norm, w,  stride=self.stride, padding=self.padding, groups=self.groups )        
        else:
            x_int, x_scale = self.activation_quant(x_norm)
            x_quant = x_norm + (x_int / x_scale - x_norm).detach()

            w_int, w_scale, _ = self.weight_quant(w)
            w_quant = w + (w_int / w_scale - w).detach()

            y = F.conv2d(x_quant, w_quant, groups=self.groups, stride=self.stride, padding=self.padding, bias=None)
        return y

    def Normalize(self, x):
            """ Normalization. Normalizes along the last dimension -> different normalization value for each activation vector.
            Args:
            x: an input tensor with shape [n, d]
            Returns:
            y: a normalized tensor with shape [n, d]
            """
            if self.NormType == 'RMS':
                y = torch.sqrt(torch.mean(x**2, dim=(-2,-1), keepdim=True))            
                z = x / y
            elif self.NormType == 'None':
                z = x
            else:
                raise AssertionError(f"Invalid NormType: {self.NormType}. Expected one of: 'RMS', 'None'")
            return z

class QuantizedModel:
    """
    This class represents a quantized model. It provides functionality to quantize a given model.
    """   
     
    def __init__(self, model = None):
        self.quantized_model=None
        self.total_bits=0

        if model is not None:
            self.quantized_model, _ = self.quantize(model)

    def totalbits(self):
        """
        Returns the total number of bits used by the quantized model.
        """
        return self.total_bits
                                      
    def quantize(self,model):
        """
        This method quantizes the weights of the given model.

        Parameters:
        model (torch.nn.Module): The PyTorch model to be quantized.
        Only the weights of the BitLinear layers are quantized.

        Returns:
        list: A list of dictionaries containing information about each layer of the quantized model.
        int: The total number of bits used by the quantized model.
        """
        quantized_model = []
        totalbits = 0
        for i, layer in enumerate(model.modules()):
            print(i, layer.__class__.__name__)
            if isinstance(layer, BitLinear):
                w         = layer.weight.data

                # print(f'layer: {layer} s:{layer.s})')
                u, scale, bpw = layer.weight_quant(w)
                numscale = 0 # TODO: store scale value for "PerOutput" scaling
                # print (scale)                

                totalbits += bpw * u.numel() + numscale * 8
                quantized_weight = u.cpu().numpy()

                layer_info = {
                    'layer_type': 'BitLinear',
                    'layer_order': i,
                    'incoming_weights': quantized_weight.shape[1],
                    'outgoing_weights': quantized_weight.shape[0],
                    'quantized_weights': quantized_weight.tolist(),
                    'WScale': layer.WScale,
                    # 'quantized_scale': scalequant.cpu().numpy().tolist() if layer.WScale=='PerOutput' else [], # TODO: "PerOutput" scaling
                    'bpw': bpw, # bits per weight
                    'quantization_type': layer.QuantType
                }
                quantized_model.append(layer_info)

            elif isinstance(layer, BitConv2d):
                w = layer.weight.data
                u, scale, bpw = layer.weight_quant(w)
                quantized_weight = u.cpu().numpy()

                totalbits += bpw * u.numel()
                layer_info = {
                    'layer_type': 'BitConv2d',
                    'layer_order': i,
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'incoming_x': 0,  # will be updated during inference
                    'incoming_y': 0,
                    'outgoing_x': 0,
                    'outgoing_y': 0,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding,
                    'groups': layer.groups,
                    'quantized_weights': quantized_weight.tolist(),
                    'bpw': bpw,
                    'quantization_type': layer.QuantType
                }
                quantized_model.append(layer_info)
                
            elif isinstance(layer, nn.MaxPool2d):
                layer_info = {
                    'layer_type': 'MaxPool2d',
                    'layer_order': i,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride
                }
                quantized_model.append(layer_info)

        self.total_bits = totalbits                
        self.quantized_model = quantized_model

        return quantized_model, totalbits 

    def inference_quantized(self, input_data):
        """
        This function performs inference on the given quantized model with the provided input data.

        Parameters:
        quantized_model (list): A list of dictionaries containing information about each layer of the quantized model.
        input_data (torch.Tensor): The input data to be used for inference.

        Returns:
        torch.Tensor: The output of the model after performing inference.
        """
        
        if not self.quantized_model:
            raise ValueError("quantized_model is empty or None")

        scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
        current_data = np.round(input_data * scale).clip(-128, 127) 

        for layer_info in self.quantized_model[:-1]:  # For all layers except the last one
            # print(f'layer: {layer_info["layer_type"], layer_info["layer_order"] }')

            if layer_info['layer_type'] == 'BitLinear':

                if len(current_data.shape) == 4:
                    # reshape from (batch_size, channels, height, width) to (batch_size, features)
                    current_data = current_data.reshape(current_data.shape[0], current_data.shape[1] * current_data.shape[2] * current_data.shape[3])

                weights = np.array(layer_info['quantized_weights']) 
                conv = np.dot(current_data, weights.T)  # Matrix multiplication

                if layer_info['WScale']=='PerOutput':
                    scale = np.array(layer_info['quantized_scale']).transpose()
                    conv = conv * scale

                max = np.maximum(conv.max(axis=-1, keepdims=True), 1e-5)
                rescale = np.exp2(np.floor(np.log2(127.0 / max))) # Emulate normalization by shift as in C inference engine
                # rescale = 127.0 / np.maximum(conv.max(axis=-1, keepdims=True), 1e-5)   # Normalize to max 1.7 range
                current_data = np.round(conv * rescale).clip(0, 127)  # Quantize the output and ReLU

            elif layer_info['layer_type'] == 'BitConv2d':

                if len(current_data.shape) == 2:
                    # Reshape from (batch_size, features) to (batch_size, channels, height, width)
                    height = width = int(np.sqrt(current_data.shape[1] // layer_info['in_channels']))
                    current_data = current_data.reshape(current_data.shape[0], layer_info['in_channels'], height, width)

                kernel_size = layer_info['kernel_size'][0]  # Assuming square kernel
                groups = layer_info['groups']
                in_channels = layer_info['in_channels']
                out_channels = layer_info['out_channels']   

                weights = np.array(layer_info['quantized_weights']).reshape(
                    out_channels, in_channels // groups, kernel_size, kernel_size)
                
                # print(f'weights: {weights.shape} data: {current_data.shape}')
                output = np.zeros((current_data.shape[0], layer_info['out_channels'],
                                current_data.shape[2] - kernel_size + 1, current_data.shape[3] - kernel_size + 1))

                # update the incoming and outgoing dimensions
                layer_info['incoming_x'] = current_data.shape[2]
                layer_info['incoming_y'] = current_data.shape[3]

                layer_info['outgoing_x'] = output.shape[2]   
                layer_info['outgoing_y'] = output.shape[3]

                for g in range(groups):
                    for i in range(output.shape[2]):
                        for j in range(output.shape[3]):
                            patch = current_data[:, g*(in_channels//groups):(g+1)*(in_channels//groups), 
                                                i:i+kernel_size, j:j+kernel_size]
                            group_weights = weights[g*(out_channels//groups):(g+1)*(out_channels//groups)]
                            output[:, g*(out_channels//groups):(g+1)*(out_channels//groups), i, j] = \
                                np.sum(patch[:, np.newaxis, :, :, :] * group_weights, axis=(2, 3, 4))
                        

                # print(f'output: {output.shape}')
                # Apply ReLU and quantize
                output = np.maximum(output, 0)
                max_val = np.max(output, axis=(1, 2, 3), keepdims=True)
                current_data = np.round(output * (127.0 / max_val)).clip(0, 127).astype(np.int8)

            elif layer_info['layer_type'] == 'MaxPool2d':
                pool_size = layer_info['kernel_size']
                stride = layer_info['stride']
                batch_size, channels, height, width = current_data.shape
                pooled_height = (height - pool_size) // stride + 1
                pooled_width = (width - pool_size) // stride + 1
                pooled_output = np.zeros((batch_size, channels, pooled_height, pooled_width), dtype=current_data.dtype)

                for i in range(pooled_height):
                    for j in range(pooled_width):
                        h_start = i * stride
                        h_end = h_start + pool_size
                        w_start = j * stride
                        w_end = w_start + pool_size
                        pooled_output[:, :, i, j] = np.max(current_data[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

                current_data = pooled_output          
                

        # no renormalization for the last layer
        weights = np.array(self.quantized_model[-1]['quantized_weights'])
        logits = np.dot(current_data, weights.T)  # Matrix multiplication

        if self.quantized_model[-1]['WScale']=='PerOutput':
            scale = np.array(self.quantized_model[-1]['quantized_scale']).transpose()
            logits = logits * scale

        return logits