import torch
import torch.nn as nn
import pandas as pd
from visualize import save_plot 
from torchinfo import summary


class KnotID(nn.Module):

	def __init__(self):
		super(KnotID, self).__init__()

		# Layers for learning features
		# input shape (1, 3, 128, 128)
		self.feature_learning = nn.Sequential(
			nn.Conv2d(3, 10, 3, 1, 1),      # (1, 10, 128, 128)
			nn.ReLU(),
			nn.MaxPool2d(2),                # (1, 10, 64, 64)
			nn.Conv2d(10, 20, 3, 1, 1),     # (1, 20, 64, 64)
			nn.ReLU(),
			nn.MaxPool2d(2),                # (1, 20, 32, 32)
		)

		# Layers for classifying images
		self.classification = nn.Sequential(
			nn.Flatten(1),                  # (1, 20480)
			nn.Linear(20480, 2048),         # (1, 2048)
			nn.ReLU(),
			nn.Linear(2048, 256),           # (1, 256)
			nn.ReLU(),
			nn.Linear(256, 10)              # (1, 10)
		)

	def forward(self, x):
		x = self.feature_learning(x)
		x = self.classification(x)
		return x

	def save(self, model_id, data, train_time):
		df = pd.DataFrame(data)
		df.to_csv(f'./models/tables/knot-id_{model_id:04}.csv')

		save_plot(
			model_id,
			data['train_losses'],
			data['test_losses'],
			data['train_accuracies'],
			data['test_accuracies'],
			data['epochs']
		)

		model_summary = str(summary(self, input_size=(1,3,128,128), verbose=0))

		with open(f'./models/summaries/knot-id_{model_id:04}.txt', 'w') as file:
			file.write(f"img_size: {data['img_size']}\n")
			file.write(f"batch_size: {data['batch_size']}\n")
			file.write(f"learning_rate: {data['learning_rate']}\n")
			file.write(f"num_epochs: {data['num_epochs']}\n\n")
			file.write(f'training time: {train_time:}\n\n')
			file.write(f'str(self)\n\n')
			file.write(model_summary)

		torch.save(self.state_dict(), f'./models/serialized/knot-id_{model_id:04}.pt')