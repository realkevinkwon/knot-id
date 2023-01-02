import torch
from model import KnotID
from torchinfo import summary

MODEL_DIR = './models'

def main():
	model_name = 'knot-id_0001'
	model = KnotID()

	model.load_state_dict(torch.load(f'{MODEL_DIR}/{model_name}.pt'))
	summary(model, input_size=(1,3,128,128))

if __name__ == '__main__':
	main()