import os
from torchvision.datasets.vision import VisionDataset
from PIL import Image


DATA_ROOT_RAW = './data/raw'                       		# Root directory for raw data
DATA_ROOT_PROCESSED = './data/processed'                # Root directory for processed data
DATA_DIR = '10Knots_128'                      			# Name of specific dataset
DATA_PATH = os.path.join(DATA_ROOT_PROCESSED, DATA_DIR)
CLASSES = [
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
		self.split = split

		super().__init__(DATA_ROOT_PROCESSED, transforms=None, transform=transform)
		
		for idx, class_name in enumerate(CLASSES):
			class_path = os.path.join(DATA_PATH, self.split, class_name)
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