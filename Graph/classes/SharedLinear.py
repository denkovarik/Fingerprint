import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class SharedLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, useBatchnorm=False, dropoutRate=0.0):
        super(SharedLinear, self).__init__()
        self.maxInFeatures = max_in_features
        self.maxOutFeatures = max_out_features
        self.useBatchnorm = useBatchnorm
        self.dropoutRate = dropoutRate

        # Initialize the weights and biases for the maximum configuration
        self.weight = nn.Parameter(torch.Tensor(self.maxOutFeatures, self.maxInFeatures))
        self.bias = nn.Parameter(torch.Tensor(self.maxOutFeatures))

        # Initialize weights using Kaiming (He) initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.maxInFeatures  # Number of input features
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        # Batch normalization layer
        #if self.useBatchnorm:
        #    self.batchnorm = nn.BatchNorm1d(maxOutFeatures)

        # Dropout layer
        #self.dropout = nn.Dropout(p=dropoutRate)


    def __str__(self):
        strRep = (f"SharedLinear("
                  f"max_in_features={self.maxInFeatures}, "
                  f"max_out_features={self.maxOutFeatures}"
                  f")")
        return strRep


    def forward(self, x, inChannels, outChannels):
        # Dynamically select the subset of weights and biases
        weight = self.weight[:outChannels, :inChannels]
        bias = self.bias[:outChannels]
        
        # Flatten input if not already flat
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        x = F.linear(x, weight, bias)

        #if self.useBatchnorm:
        #    x = self.batchnorm(x)

        #x = self.dropout(x)
        
        return x
