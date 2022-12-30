from dataset import CLASSES, DATA_RAW, DATA_PROCESSED
import os
import math
import random
from torchvision.datasets.vision import VisionDataset
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image

TEST_SPLIT = 0.2
VAL_SPLIT = 0.0
TRAIN_SPLIT = 1.0 - TEST_SPLIT - VAL_SPLIT

LIGHTINGS = [
    'DiffuseLight',
    'SourceLight-Above',
    'SourceLight-Side'
]

LOOSENESS_LIST = [
    'Loose',
    'Set',
    'VeryLoose'
]

class ProcessKnots(VisionDataset):

    def __init__(self, class_name, lighting, looseness, class_idx, transform=None):
        self.transform = transform
        self.filepaths = []
        self.filenames = []
        self.targets = []

        super().__init__(DATA_RAW, transforms=None, transform=transform)

        path = os.path.join(DATA_RAW, '10Knots', class_name, lighting, looseness)
        for file in os.listdir(path):
            if file != '.DS_Store':
                self.filepaths.append(os.path.join(path, file))
                self.filenames.append(file)
                self.targets.append(class_idx)

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_filename(self, idx):
        return self.filenames[idx]


def process_images(img_size):
    img_dir = f'10Knots_{img_size}'
    dir_list = os.listdir(DATA_PROCESSED)

    if img_dir in dir_list:
        print(f'{img_dir} already exists')
        return

    transform = transforms.Compose([
        transforms.CenterCrop(3456),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_path = os.path.join(DATA_PROCESSED, img_dir, 'train')
    test_path = os.path.join(DATA_PROCESSED, img_dir, 'test')
    val_path = os.path.join(DATA_PROCESSED, img_dir, 'val')
    os.makedirs(train_path, 0o755)
    os.makedirs(test_path, 0o755)
    os.makedirs(val_path, 0o755)

    for class_idx, class_name in enumerate(CLASSES):
        print(f'Saving {class_name!r} ... ', end='')

        os.mkdir(os.path.join(train_path, class_name), 0o755)
        os.mkdir(os.path.join(test_path, class_name), 0o755)
        os.mkdir(os.path.join(val_path, class_name), 0o755)

        for lighting in LIGHTINGS:
            for looseness in LOOSENESS_LIST: 
        
                img_data = ProcessKnots(class_name, lighting, looseness, class_idx, transform=transform)

                indices = list(range(len(img_data)))
                random.shuffle(indices)

                train_start = 0
                train_end = round(TRAIN_SPLIT * len(indices))
                test_start = train_end
                test_end = test_start + math.floor(TEST_SPLIT * len(indices))
                val_start = test_end
                val_end = len(indices)

                train_indices = indices[train_start:train_end]
                test_indices = indices[test_start:test_end]
                val_indices = indices[val_start:val_end]

                for curr_split, curr_indices in zip(['train','test','val'],[train_indices,test_indices,val_indices]):
                    for idx in curr_indices:
                        img, _ = img_data.__getitem__(idx)
                        img_path = os.path.join(DATA_PROCESSED, img_dir, curr_split, class_name, img_data.get_filename(idx))
                        save_image(img, img_path)
        
        print('Done')


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

    process_images(img_size)
