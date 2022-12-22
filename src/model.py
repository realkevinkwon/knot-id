import torch
import torch.nn as nn

class KnotClassifier(nn.Module):

    def __init__(self):
        super(KnotClassifier, self).__init__()

    def forward(self, x):
        return x