from model import KnotClassifier
from dataset import Knots
from torchvision import transforms

def main():
    transform = transforms.Compose([transforms.CenterCrop(3456), transforms.Resize((256,256))])

    knots_data = Knots(transform=transform)
    img, target = knots_data.__getitem__(683)
    img.show()

if __name__ == '__main__':
    main()