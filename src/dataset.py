import os
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class Knots(VisionDataset):

    def __init__(self, transform=None):
        self.root = './data/'
        self.transform = transform
        self.filepaths = []
        self.targets = []
        self.classes = {}

        super().__init__(self.root, transforms=None, transform=transform)

        class_idx = 0
        for filename in os.listdir(os.path.join(self.root, '10Knots')):
            if filename != '.DS_Store':
                self.classes[class_idx] = filename
                class_idx += 1
        
        for idx, label in self.classes.items():
            for path, _, filenames in os.walk(os.path.join(self.root, '10Knots', label)):
                for filename in filenames:
                    if filename != '.DS_Store':
                        self.filepaths.append(os.path.join(path, filename))
                        self.targets.append(idx)

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_class(self, idx):
        return self.classes[idx]