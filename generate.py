import os
import argparse

import matplotlib.pyplot as plt
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.glow import Glow as Glow_uncond
from utils.glow_cond import Glow as Glow_cond

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_samples(samples, samples_per_row=5):
	# Plotting function
	n_rows = len(samples) // samples_per_row
	n_tot = int(n_rows*samples_per_row)
	samples = samples[:n_tot]
	fig = plt.figure(figsize=(2*samples_per_row, 2*n_rows))
	for i, out in enumerate(samples):
		a = fig.add_subplot(n_rows, samples_per_row, i+1)
		out_sh = out.permute(1,2,0)
		plt.imshow(out_sh, cmap='gray')
		a.axis("off")

	plt.show()

def generate(args):
	data_shape=(1,28,28) # Change if model was trained on data with different dimensions
	num_labels = 10 # Change if model was trained on data with different number of labels
	if (args.conditional==True):
		model = Glow_cond(data_shape, args.hidden_channels, args.num_flow_steps, args.ACL_layers, args.num_levels, args.num_splits, num_labels, device, args.chunk_size)
		model.load_state_dict(torch.load(args.model_path), strict=False)
		model.eval()
		samples = model.sample(torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4]).to(device), args.num_samples).cpu().detach()
	else:
		model = Glow_uncond(data_shape, args.hidden_channels, args.num_flow_steps, args.ACL_layers, args.num_levels, args.num_splits, device, args.chunk_size)
		model.load_state_dict(torch.load(args.model_path))
		model.eval()
		samples = model.sample(args.num_samples).cpu().detach()
	
	plot_samples(samples)

