import torch.nn as nn
import torch.nn.functional as F

class KnotClassifier(nn.Module):

    def __init__(self):
        super(KnotClassifier, self).__init__()

        # input shape (64, 3, 128, 128)
        self.feature_learning = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, 1),      # (64, 10, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (64, 10, 64, 64)
            nn.Conv2d(10, 20, 3, 1, 1),     # (64, 20, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (64, 20, 32, 32)
        )

        self.classification = nn.Sequential(
            nn.Flatten(1),                  # (64, 20480)
            nn.Linear(20480, 2048),         # (64, 2048)
            nn.ReLU(),
            nn.Linear(2048, 256),           # (64, 256)
            nn.ReLU(),
            nn.Linear(256, 10)              # (64, 10)
        )

    def forward(self, x):
        x = self.feature_learning(x)
        x = self.classification(x)
        return x