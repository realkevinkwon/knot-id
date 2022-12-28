from models.model import KnotClassifier
from data.dataset import Knots
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Constants
batch_size = 1                      # Batch size
img_size = 128                      # Size of the images in the dataset
num_epochs = 5                      # Number of iterations for training
learning_rate = 1e-4                # Learning rate
classes = [                         # List of classes as strings
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
num_classes = len(classes)          # Number of classes in the datsets


def main():
	# Transforms to be applied to the dataset
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.7019, 0.4425, 0.1954), (0.1720, 0.1403, 0.1065))
	])

	# Initialize datasets
	train_data = Knots(split='train', transform=transform)
	test_data = Knots(split='test', transform=transform)

	# Create data loaders for training and testing
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	# Instantiate model, loss function, and optimizer
	model = KnotClassifier()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# Train model for 'num_epochs' epochs
	for epoch in range(1, num_epochs+1):
		train(model, train_loader, loss_fn, optimizer, epoch)
		test(model, test_loader, loss_fn, epoch)


# Function to train the model
def train(model, train_loader, loss_fn, optimizer, epoch):
	model.train()       # Set model to train

	# Iterate through train dataset, 'batch_size' images at a time
	for batch_idx, (images, targets) in enumerate(train_loader):
		optimizer.zero_grad()               # Zero the gradients
		output = model(images)              # Predict class
		loss = loss_fn(output, targets)     # Compute loss
		loss.backward()                     # Compute gradient
		optimizer.step()                    # Update parameters

		# Print training loss every 100 images
		if batch_idx % 100 == 0:
			print(
				f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}] '
				f'Loss: {loss.item():.8f}'
			)


# Function to test the model
def test(model, test_loader, loss_fn, epoch):
	model.eval()        # Set model to evaluate
	test_loss = 0       # Initialize test loss
	correct = 0         # Initialize number of correct predictions
	
	with torch.no_grad():       # Make sure PyTorch does not update the gradients
		# Iterate through test dataset
		for images, targets in test_loader:
			output = model(images)                                  # Predict class
			test_loss += loss_fn(output, targets).item()            # Compute loss
			pred = output.data.max(1, keepdim=True)[1]              # Get the prediction
			correct += pred.eq(targets.data.view_as(pred)).sum()    # Add to total correct predictions

		test_loss /= len(test_loader.dataset)       # Calculate average test loss
		print(                                      # Print average test loss and prediction accuracy
			f'Test result on epoch {epoch}: '
			f'Avg loss is {test_loss:.8f}, '
			f'Accuracy: {(100.0 * correct / len(test_loader.dataset)):.2f}%'
		)
	

if __name__ == '__main__':
	main()