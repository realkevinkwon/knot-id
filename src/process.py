from dataset import CLASSES, DATA_RAW, DATA_PROCESSED
import os
import math
import random
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


def process_images(img_size):
	img_dir = f'10Knots_{img_size}'
	dir_list = os.listdir(DATA_PROCESSED)

	if img_dir in dir_list:
		print(f'{img_dir} already exists')
		return

	train_path = os.path.join(DATA_PROCESSED, img_dir, 'train')
	test_path = os.path.join(DATA_PROCESSED, img_dir, 'test')
	val_path = os.path.join(DATA_PROCESSED, img_dir, 'val')
	os.makedirs(train_path, 0o755)
	os.makedirs(test_path, 0o755)
	os.makedirs(val_path, 0o755)

	for class_name in CLASSES:
		print(f'Saving {class_name!r} ... ', end='')

		os.mkdir(os.path.join(train_path, class_name), 0o755)
		os.mkdir(os.path.join(test_path, class_name), 0o755)
		os.mkdir(os.path.join(val_path, class_name), 0o755)

		for lighting in LIGHTINGS:
			for looseness in LOOSENESS_LIST: 
				img_root = os.path.join(DATA_RAW, '10Knots', class_name, lighting, looseness)
				img_files = [file for file in os.listdir(img_root) if file != '.DS_Store']
				img_paths = [os.path.join(img_root, file) for file in img_files]
		
				indices = list(range(len(img_files)))
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
						img = Image.open(img_paths[idx])
						img.crop((864, 0, 4320, 3456))
						img.resize((img_size, img_size))
						img.save(os.path.join(
							DATA_PROCESSED,
							img_dir,
							curr_split,
							class_name,
							img_files[idx]
						))
		
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