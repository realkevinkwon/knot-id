import os
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class Knots(VisionDataset):

    def __init__(self, transform=None):
        self.root = './data/'
        self.transform = transform
        self.filepaths = []
        self.targets = []

        super().__init__(self.root, transforms=None, transform=transform)

        classes = []
        for item in os.listdir(os.path.join(self.root, '10Knots')):
            if item != '.DS_Store':
                classes.append(item)
        
        for label in classes:
            for path, _, filenames in os.walk(os.path.join(self.root, '10Knots', label)):
                for filename in filenames:
                    if filename != '.DS_Store':
                        self.filepaths.append(os.path.join(path, filename))
                        self.targets.append(label)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target