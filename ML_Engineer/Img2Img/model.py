# Author: Keanu Spies (keanuspies@gmail.com)
import torch.nn as nn
import torch

class DoubleFilter(nn.Module):
    """
    A model that applies two filters and outputs the normalized sum of these filters. 
    """
    def __init__(self, normalization=True):
        super().__init__()
        self.conv_layer = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_layer(x)
        # normalization (sqrt(x^2 + y^2))
        out = torch.mul(out, out)
        out = torch.sum(out, axis =1)
        out = torch.sqrt(out)
        return out.unsqueeze(1) # quick fix for loss of channel

class SingleFilter(nn.Module):
    """
    A model that applies a filter and normalizes the output of the filter or not.  
    """

    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv_layer(x)
        # normalization (sqrt(x^2))
        out = torch.abs(out)
        return out 


class SingleFilterNoNorm(nn.Module):
    """
    A model that applies a filter without normalization. 
    Quick Fix: This had to be its own class because nn doesn't have a Abs() module anymore.
                likely a smarter wat to incorporate the two classes into one, but I decided
                to just avoid that effort. 
    """
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv_layer(x)
        return out 