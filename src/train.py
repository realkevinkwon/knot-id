import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import KnotID
from dataset import Knots
from torchvision import transforms
from torch.utils.data import DataLoader


batch_size = 1
img_size = 128
num_epochs = 5
learning_rate = 1e-4
model_dir = './models'				# Location of serialized models
model_id = 3


def main():
	train_loader, test_loader = prepare_data()

	model = KnotID()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	train_losses, test_losses = [], []
	train_accuracies, test_accuracies = [], []
	epochs = range(1, num_epochs+1)

	train_start = time.time()
	for epoch in epochs:
		train(model, train_loader, loss_fn, optimizer, epoch)

		print('Testing model on train dataset')
		loss, accuracy = test(model, train_loader, loss_fn, epoch)
		train_losses.append(loss)
		train_accuracies.append(accuracy)

		print('Testing model on test dataset')
		loss, accuracy = test(model, test_loader, loss_fn, epoch)
		test_losses.append(loss)
		test_accuracies.append(accuracy)
	train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start))

	data = {
		'batch_size': batch_size,
		'img_size': img_size,
		'num_epochs': num_epochs,
		'learning_rate': learning_rate,
		'epochs': epochs,
		'train_losses': train_losses,
		'test_losses': test_losses,
		'train_accuracies': train_accuracies,
		'test_accuracies': test_accuracies
	}
	model.save(model_id, data, train_time)


def prepare_data():
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.7019, 0.4425, 0.1954), (0.1720, 0.1403, 0.1065))
	])

	train_data = Knots(split='train', transform=transform)
	test_data = Knots(split='test', transform=transform)

	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	return train_loader, test_loader

def train(model, train_loader, loss_fn, optimizer, epoch):
	model.train()

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


def test(model, test_loader, loss_fn, epoch):
	model.eval()
	test_loss = 0
	test_accuracy = 0
	correct = 0
	
	with torch.no_grad():       # Make sure PyTorch does not update the gradients
		for images, targets in test_loader:
			output = model(images)                                  # Predict class
			test_loss += loss_fn(output, targets).item()            # Compute loss
			pred = output.data.max(1, keepdim=True)[1]              # Get the prediction
			correct += pred.eq(targets.data.view_as(pred)).sum()    # Add to total correct predictions

		test_loss /= len(test_loader.dataset)       # Calculate average test loss
		test_accuracy = 100.0 * correct / len(test_loader.dataset)
		print(                                      # Print average test loss and prediction accuracy
			f'Test result on epoch {epoch}: '
			f'Avg loss is {test_loss:.8f}, '
			f'Accuracy: {test_accuracy:.2f}%'
		)
	
	return test_loss, test_accuracy
	

if __name__ == '__main__':
	main()