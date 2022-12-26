import os
from torchvision.datasets.vision import VisionDataset
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image

data_root = './data'
data_dir = '10Knots_128'
data_path = os.path.join(data_root, data_dir)
classes = [
    'Alpine Butterfly Knot',
    'Bowline Knot',
    'Clove Hitch',
    'Figure-8 Knot',
    'Figure-8 Loop',
    'Fisherman\'s Knot',
    'Flemish Bend',
    'Overhand Knot',
    'Reef Knot',
    'Slip Knot'
]

class Knots(VisionDataset):

    def __init__(self, split, transform=None):
        self.transform = transform
        self.filepaths = []
        self.targets = []
        self.classes = {}
        self.split = split

        super().__init__(data_root, transforms=None, transform=transform)
        
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(data_path, self.split, class_name)
            for file in os.listdir(class_path):
                if file != '.DS_Store':
                    self.filepaths.append(os.path.join(class_path, file))
                    self.targets.append(idx)

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target