import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicNet(nn.Module):
    def __init__(self, input_shape, nUnits, nLayers):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.nLayers = nLayers
        self.layers['input'] = nn.Linear(input_shape, nUnits)
        for i in range(nLayers):
            self.layers[f'hidden_{i}'] = nn.Linear(nUnits, nUnits)
        self.layers['output'] = nn.Linear(nUnits, 1)
    def forward(self, x):
        x = F.relu(self.layers['input'](x))
        for i in range(self.nLayers):
            x = F.relu(self.layers[f'hidden_{i}'](x))
        x = self.layers['output'](x)
        return x
