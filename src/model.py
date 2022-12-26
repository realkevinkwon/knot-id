import torch.nn as nn

class KnotClassifier(nn.Module):

    def __init__(self):
        super(KnotClassifier, self).__init__()

        # input shape (64, 3, 512, 512)
        self.feature_learning = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, 1),      # (64, 10, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (64, 10, 256, 256)
            nn.Conv2d(10, 20, 3, 1, 1),     # (64, 20, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (64, 20, 128, 128)
        )

        self.classification = nn.Sequential(
            nn.Flatten(1),                  # (64, 327680)
            nn.Linear(327680, 32768),       # (64, 32768)
            nn.ReLU(),
            nn.Linear(32768, 512),          # (64, 512)
            nn.ReLU(),
            nn.Linear(512,10)                # (64, 10)
        )

    def forward(self, x):
        x = self.feature_learning(x)
        x = self.classification(x)
        return x