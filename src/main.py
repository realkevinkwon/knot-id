from model import KnotClassifier
from dataset import Knots
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 64
img_size = 32
num_classes = 10
num_epochs = 1
test_split = 0.2
train_split = 1.0 - test_split
learning_rate = 0.01

def main():
    transform = transforms.Compose([
        transforms.CenterCrop(3456),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.7019, 0.4425, 0.1954), (0.1720, 0.1403, 0.1065))
    ])

    train_data, test_data = random_split(Knots(transform=transform), [train_split, test_split])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = KnotClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        train(model, train_loader, loss_fn, optimizer, epoch)
        test(model, test_loader, loss_fn, epoch)


def train(model, train_loader, loss_fn, optimizer, epoch):
    model.train()

    for batch_idx, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}]'
                f'Loss: {loss.item():.4f}'
            )

def test(model, test_loader, loss_fn, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            output = model(images)
            test_loss += loss_fn(output, targets, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            print(
                f'Test result on epoch {epoch}: '
                f'Avg loss is {test_loss:.4f}, '
                f'Accuracy: {(100.0 * correct / len(test_loader.dataset)):.2f}%'
            )


if __name__ == '__main__':
    main()