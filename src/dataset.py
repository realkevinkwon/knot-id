import os
from torchvision.datasets.vision import VisionDataset
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image

data_root = './data'
data_dir = '10Knots_32'
data_path = os.path.join(data_root, data_dir)

class Knots(VisionDataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.filepaths = []
        self.targets = []
        self.classes = {}

        super().__init__(data_root, transforms=None, transform=transform)

        class_idx = 0
        for filename in os.listdir(data_path):
            if filename != '.DS_Store':
                self.classes[class_idx] = filename
                class_idx += 1
        
        for idx, label in self.classes.items():
            for class_path, _, filenames in os.walk(os.path.join(data_path, label)):
                for filename in filenames:
                    if filename != '.DS_Store':
                        self.filepaths.append(os.path.join(class_path, filename))
                        self.targets.append(idx)

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_class(self, target):
        return self.classes[target]