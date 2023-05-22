import argparse
import torch
import torch.nn as nn
import math

from models.barlowtwins import BarlowTwins, BarlowLoss
from optimizer.lars import Lars
from transforms.barlow_transforms import BarlowTransform
import torchvision

# parser = argparse.ArgumentParser()
# parser.add_argument('data')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def scheduler(step, optimizer,epochs, loader, num_warmup,start_lr, end_lr, lr_weights, lr_bias):
	warmup_steps = len(loader) * num_warmup
	max_steps = len(loader) * epochs
	if step < warmup_steps:
		lr =  start_lr * step/warmup_steps
	else:
		step -= warmup_steps
		max_steps -= warmup_steps
		q = 0.5 * (1 + math.cos(math.pi * step/max_steps))
		lr = start_lr * q + end_lr * (1-q)
	
	optimizer.param_groups[0]['lr'] = lr * lr_weights
	optimizer.param_groups[1]['lr'] = lr * lr_bias
	


def main():
	# args = parser.parse_args()

	backbone = torchvision.models.resnet18(zero_init_residual=True)
	backbone.fc = nn.Identity()
	projector = nn.Sequential(nn.Linear(512, 1024),
								nn.BatchNorm1d(1024),
								nn.ReLU(inplace = True),
								nn.Linear(1024, 1024),
								nn.BatchNorm1d(1024),
								nn.ReLU(inplace = True),
								nn.Linear(1024, 1024),
								nn.BatchNorm1d(1024)
								)
	model = BarlowTwins(backbone, projector)

	loss = lambda x,y: BarlowLoss(x,y, w = 0.0051)

	param_weights = []
	param_biases = []
	for param in model.parameters():
		if param.ndim == 1:
			param_biases.append(param)
		else:
			param_weights.append(param)
	parameters = [{'params': param_weights}, {'params': param_biases}]

	optimizer = Lars(parameters, lr = 0, weight_decay = 1e-6, momentum = 0.9)

	# dataset = torchvision.datasets.ImageFolder('data', BarlowTransform())
	dataset = torchvision.datasets.MNIST('mnist', download = True, transform= BarlowTransform(img_size = 28, to_rgb = True))
	schedul = lambda step, optimizer: scheduler(step, optimizer, epochs= 20, loader = dataset, num_warmup=10, start_lr = 8, end_lr=0.08, lr_weights=0.2, lr_bias = 0.0048)
	train(model, optimizer, loss, dataset,scheduler = schedul ,epochs = 20, lr = 0.0, batch_size = 64, num_workers = 1)



def train(model, optimizer, loss_function, dataset, scheduler, epochs, lr, batch_size, num_workers, checkpoint = None):

	loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, num_workers=num_workers, shuffle = True)
	model.train()
	print(f'Start training...')
	for e in range(epochs):
		print(f'Epoch {e}:')
		loss = 0
		for i, ((x1, x2), _) in enumerate(loader):
			
			x1 = x1.to(DEVICE)
			x2 = x2.to(DEVICE)
			z1, z2 = model(x1,x2)
			loss += loss_function(z1,z2)

			# simulate a higher batch_size
			if (i * 8)  % batch_size == 0:
				scheduler(i, optimizer)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				print(f'Batch_loss: {loss/batch_size}')
				loss = 0







if __name__ == "__main__":
	main()