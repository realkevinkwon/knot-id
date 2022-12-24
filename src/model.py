import torch.nn as nn

class KnotClassifier(nn.Module):

    def __init__(self):
        super(KnotClassifier, self).__init__()

        # input shape (64, 3, 32, 32)
        self.feature_learning = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, 1),      # (64, 10, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (64, 10, 16, 16)
            nn.Conv2d(10, 20, 3, 1, 1),     # (64, 20, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (64, 20, 8, 8)
        )

        self.classification = nn.Sequential(
            nn.Flatten(1),                  # (64, 1280)
            nn.Linear(1280, 256),           # (64, 256)
            nn.ReLU(),
            nn.Linear(256,64),              # (64, 64)
            nn.ReLU(),
            nn.Linear(64,10)                # (64, 10)
        )

    def forward(self, x):
        x = self.feature_learning(x)
        x = self.classification(x)
        return x