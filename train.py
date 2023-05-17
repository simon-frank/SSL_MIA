import argparse
import torch
import torchvision
import torch.nn as nn

from models.barlowtwins import BarlowTwins, BarlowLoss
from optimizer.lars import Lars
from transforms.barlow_transforms import BarlowTransform

# parser = argparse.ArgumentParser()
# parser.add_argument('data')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
	# args = parser.parse_args()

	backbone = torchvision.models.resnet18(zero_init_residual=True)
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

	optimizer = Lars(model.parameters(), lr = 0.0007, weight_decay = 1e-6, momentum = 0.9)

	dataset = torchvision.datasets.ImageFolder('data', BarlowTransform())

	train(model, optimizer, loss, dataset, epochs = 10, lr = 0.0, batch_size = 16, num_workers = 1)


def train(model, optimizer, loss, dataset, epochs, lr, batch_size, num_workers, checkpoint = None):

	loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)
	model.train()

	for e in range(epochs):
		for i, ((x1, x2), _) in enumerate(loader):
			x1 = x1.to(DEVICE)
			x2 = x2.to(DEVICE)
			optimizer.zero_grad()

			z1, z2 = model(x1,x2)

			# print(z1,z2)

			l = loss(z1,z2)
			l.backward()
			optimizer.step()

			print(l)







if __name__ == "__main__":
	main()