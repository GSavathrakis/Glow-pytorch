from tqdm import tqdm 
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Conv2d

from torch.optim import AdamW

import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from utils.glow import Glow

parser = argparse.ArgumentParser('Glow Normalizing Flow model', add_help=False)
parser.add_argument('--dataset', type=str, default='MNIST')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_flow_steps', type=int, default=2)
parser.add_argument('--num_levels', type=int, default=2)
parser.add_argument('--num_splits', type=int, default=2)


args = parser.parse_args() 

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

def squeeze(x, chunk_size=2):
	c = x.shape[1]
	h = x.shape[2]
	w = x.shape[3]
	y = x.permute(0,2,3,1)
	y = y.view(x.shape[0], h//chunk_size, chunk_size, w//chunk_size, chunk_size, c)
	y = y.permute(0,1,3,2,4,5)
	y = y.reshape(x.shape[0], h//chunk_size, w//chunk_size, (chunk_size**2)*c)
	y = y.permute(0,3,1,2)
	return y

def unsqueeze(x, chunk_size=2):
	c_pr = x.shape[1]
	h_pr = x.shape[2]
	w_pr = x.shape[3]

	y = x.permute(0,2,3,1)
	y = y.view(x.shape[0], h_pr*chunk_size, chunk_size, w_pr*chunk_size, chunk_size, c_pr)
	y = y.permute(0,1,3,2,4,5)
	y = y.reshape(x.shape[0], h_pr*chunk_size, w_pr*chunk_size, c//(chunk_size**2))
	y = y.permute(0,3,1,2)
	return y

def main(args):

	train_dataset = getattr(torchvision.datasets, args.dataset)(root='./data', train=True, download=True, transform=ToTensor())
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

	if not os.path.isdir('saved_models'):
		os.makedirs('saved_models')

	Iter = iter(train_loader)
	images, _ = next(Iter)
	im_sq = squeeze(images[0].reshape(1,*images[0].shape))
	data_shape = im_sq.detach().numpy().shape
	data_shape = list(data_shape)
	data_shape.pop(0)
	data_shape = tuple(data_shape)

	model = Glow(data_shape, args.hidden_channels, args.num_flow_steps, args.num_levels, args.num_splits, device).to(device)
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	for epoch in range(args.epochs):
		tot_log_likelihood = 0
		batch_counter = 0
		for i, (x,y) in enumerate(tqdm(train_loader)):
			model.zero_grad()

			x = x.to(device)
			y = y.to(device)

			x_1 = squeeze(x)
			z, z_disc, log_likelihood = model(x_1, i)
			loss = -torch.mean(log_likelihood)  # NLL

			loss.backward()
			optimizer.step()		  

			tot_log_likelihood -= loss
			batch_counter += 1
			break

		mean_log_likelihood = tot_log_likelihood / batch_counter  # normalize w.r.t. the batches
		print(f'Epoch {epoch+1:d} completed. Log Likelihood: {mean_log_likelihood:.4f}')
		break

	#torch.save(model.state_dict(), f'saved_models/glow.pt')


if __name__ == '__main__':
	main(args)