from torch.utils.data import Dataset

class Knots(Dataset):

    def __init__(self, root, train, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target