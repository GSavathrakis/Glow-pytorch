import os
import argparse

import matplotlib.pyplot as plt
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.glow import Glow

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_samples(samples):
	fig, ax = plt.subplots(1,5, figsize=(12,9))

	n_samp=0
	for samp in samples:
		shape = samp.shape
		#print(samp.shape)
		# Restrict pixel values from 0 to 1
		#arr_nump = torch.clamp(samp, 0, 1).cpu().detach().numpy().copy()
		ax[n_samp].imshow(samp[0], cmap='gray')
		ax[n_samp].axis('off')
		n_samp+=1

	plt.show()

def generate(args):
	data_shape=(1,28,28)
	model = Glow(data_shape, args.hidden_channels, args.num_flow_steps, args.num_levels, args.num_splits, device, args.chunk_size)
	model.load_state_dict(torch.load(args.model_path))
	model.eval()

	samples = model.sample(args.num_samples).cpu().detach().numpy()
	plot_samples(samples)

