from tqdm import tqdm 
import os
#import argparse

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

'''
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
'''

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

def train(args):

	train_dataset = getattr(torchvision.datasets, args.dataset)(root='./data', train=True, download=True, transform=ToTensor())
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

	if not os.path.isdir('saved_models'):
		os.makedirs('saved_models')

	Iter = iter(train_loader)
	images, _ = next(Iter)
	data_shape = images[0].detach().numpy().shape

	model = Glow(data_shape, args.hidden_channels, args.num_flow_steps, args.num_levels, args.num_splits, device, args.chunk_size).to(device)
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	for epoch in range(args.epochs):
		tot_log_likelihood = 0
		batch_counter = 0
		for i, (x,y) in enumerate(tqdm(train_loader)):
			
			model.zero_grad()

			x = x.to(device)
			y = y.to(device)

			z, z_disc, log_likelihood = model(x)
			loss = -torch.mean(log_likelihood)  # NLL

			loss.backward()
			optimizer.step()		  

			tot_log_likelihood -= loss
			batch_counter += 1

		mean_log_likelihood = tot_log_likelihood / batch_counter  # normalize w.r.t. the batches
		print(f'Epoch {epoch+1:d} completed. Log Likelihood: {mean_log_likelihood:.4f}')

	torch.save(model.state_dict(), f'saved_models/glow.pt')

'''
if __name__ == '__main__':
	main(args)
'''