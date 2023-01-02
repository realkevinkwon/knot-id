import torch
import torch.nn as nn
import pandas as pd
from visualize import save_plot 


class KnotID(nn.Module):

	def __init__(self):
		super(KnotID, self).__init__()

		# Layers for learning features
		# input shape (64, 3, 128, 128)
		self.feature_learning = nn.Sequential(
			nn.Conv2d(3, 10, 3, 1, 1),      # (64, 10, 128, 128)
			nn.ReLU(),
			nn.MaxPool2d(2),                # (64, 10, 64, 64)
			nn.Conv2d(10, 20, 3, 1, 1),     # (64, 20, 64, 64)
			nn.ReLU(),
			nn.MaxPool2d(2),                # (64, 20, 32, 32)
		)

		# Layers for classifying images
		self.classification = nn.Sequential(
			nn.Flatten(1),                  # (64, 20480)
			nn.Linear(20480, 2048),         # (64, 2048)
			nn.ReLU(),
			nn.Linear(2048, 256),           # (64, 256)
			nn.ReLU(),
			nn.Linear(256, 10)              # (64, 10)
		)

	def forward(self, x):
		x = self.feature_learning(x)
		x = self.classification(x)
		return x

	def save(self, model_id, data):
		df = pd.DataFrame(data)
		df.to_csv(f'./reports/tables/knot-id_{model_id}.csv')

		save_plot(
			model_id,
			data['train_losses'],
			data['test_losses'],
			data['train_accuracies'],
			data['test_accuracies'],
			data['epochs']
		)

		torch.save(self.state_dict(), f'./models/knot-id_{model_id}.pt')