import os
from torchvision.datasets.vision import VisionDataset
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image

root = './data'

class Knots(VisionDataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.filepaths = []
        self.targets = []
        self.classes = {}

        super().__init__(root, transforms=None, transform=transform)

        class_idx = 0
        for filename in os.listdir(os.path.join(root, '10Knots')):
            if filename != '.DS_Store':
                self.classes[class_idx] = filename
                class_idx += 1
        
        for idx, label in self.classes.items():
            for path, _, filenames in os.walk(os.path.join(root, '10Knots', label)):
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

    def get_class(self, target):
        return self.classes[target]

def export_images(img_size):
    img_dir = f'10Knots_{img_size}'
    dir_list = os.listdir(root)

    if img_dir in dir_list:
        return

    transform = transforms.Compose([
        transforms.CenterCrop(3456),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    img_data = Knots(transform=transform)

    os.mkdir(os.path.join(root, img_dir), 0o755)

    for idx, class_name in img_data.classes.items():
        os.mkdir(os.path.join(root, img_dir, class_name), 0o755)

    for idx in range(len(img_data)):
        img, target = img_data.__getitem__(idx)
        img_class = img_data.get_class(target)

        img_path = os.path.join(root, img_dir, img_class, f'IMG_{idx}.jpg')

        save_image(img, img_path)

        if idx % 100 == 0:
            print(f'images saved: [{idx}/{len(img_data)}]')

if __name__ == '__main__':
    img_size = 32

    while True:
        user_input = input('Enter export image size: ')

        if not user_input.isdigit():
            print('Input must be an integer')
            continue

        img_size = int(user_input)
        if img_size < 32 or img_size >= 3456:
            print('Input must be in the range [32,3456)')
            continue

        break

    export_images(img_size)