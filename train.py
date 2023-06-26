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

from utils.glow import Glow as Glow_uncond
from utils.glow_cond import Glow as Glow_cond

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

def rgb_to_gray(image):
	coef = torch.cat((0.2989*torch.ones(image.shape[0],1,image.shape[2],image.shape[3]), 0.5870*torch.ones(image.shape[0],1,image.shape[2],image.shape[3]),0.1140*torch.ones(image.shape[0],1,image.shape[2],image.shape[3])), dim=1)
	gray_image = coef*image
	gray_image = gray_image.sum(dim=1).reshape(image.shape[0], 1, image.shape[2], image.shape[3])

	return gray_image

def train(args):

	train_dataset = getattr(torchvision.datasets, args.dataset)(root='./data', train=True, download=True, transform=ToTensor())
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

	if not os.path.isdir(f'saved_models_{args.dataset}'):
		os.makedirs(f'saved_models_{args.dataset}')

	Iter = iter(train_loader)
	images, _ = next(Iter)
	if (type(train_dataset.targets)==list):
		n_labels = np.unique(np.array(train_dataset.targets)).shape[0]
	else:
		n_labels = train_dataset.targets.unique().shape[0]
	data_shape = images[0].detach().numpy().shape

	if (args.conditional==False):
		if (data_shape[0]==3):
			data_shape_new = (1, data_shape[1], data_shape[2])
			model = Glow_uncond(data_shape_new, args.hidden_channels, args.num_flow_steps, args.ACL_layers, args.num_levels, args.num_splits, device, args.chunk_size).to(device)
		else:
			model = Glow_uncond(data_shape, args.hidden_channels, args.num_flow_steps, args.ACL_layers, args.num_levels, args.num_splits, device, args.chunk_size).to(device)
	else:
		if (data_shape[0]==3):
			data_shape_new = (1, data_shape[1], data_shape[2])
			model = Glow_cond(data_shape_new, args.hidden_channels, args.num_flow_steps, args.ACL_layers, args.num_levels, args.num_splits, n_labels, device, args.chunk_size).to(device)
		else:
			model = Glow_cond(data_shape, args.hidden_channels, args.num_flow_steps, args.ACL_layers, args.num_levels, args.num_splits, n_labels, device, args.chunk_size).to(device)
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	
	for epoch in range(args.epochs):
		tot_log_likelihood = 0
		batch_counter = 0
		for i, (x,y) in enumerate(tqdm(train_loader)):
			model.zero_grad()
			if (data_shape[0]==3):
				x = rgb_to_gray(x.clone())


			x = x.to(device)
			y = y.to(device)
			if (args.conditional==True):
				z, log_likelihood = model(x, y)
			else:
				z, log_likelihood = model(x)
			loss = -torch.mean(log_likelihood)  # NLL

			loss.backward()
			optimizer.step()		  

			tot_log_likelihood -= loss
			batch_counter += 1

		
		mean_log_likelihood = tot_log_likelihood / batch_counter  # normalize w.r.t. the batches
		print(f'Epoch {epoch+1:d} completed. Log Likelihood: {mean_log_likelihood:.4f}')
		
		if (args.conditional==True):
			torch.save(model.state_dict(), f'saved_models_{args.dataset}/glow_cond_epoch_{epoch+1}.pt')
		else:
			torch.save(model.state_dict(), f'saved_models_{args.dataset}/glow_epoch_{epoch+1}.pt')
		
	#torch.save(model.state_dict(), f'saved_models_{args.dataset}/RealNVP_epoch_10.pt')