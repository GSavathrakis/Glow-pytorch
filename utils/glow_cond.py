import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, Distribution

import random
import copy
from math import prod

from itertools import permutations

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		torch.nn.init.normal_(m.weight, 1.0, 0.02)
		torch.nn.init.zeros_(m.bias)

class LogisticDistribution(Distribution):
	def __init__(self):
		super().__init__()

	def log_prob(self, x):
		return -(F.softplus(x) + F.softplus(-x))

	def sample(self, size):
		z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)

		return torch.log(z) - torch.log(1. - z)



class Actnorm(nn.Module):
	def __init__(self, input_shape):
		super(Actnorm, self).__init__()
		self.input_shape = input_shape
		self.Weight = nn.Parameter(torch.Tensor(input_shape[0],1,1))
		self.Bias = nn.Parameter(torch.Tensor(input_shape[0],1,1))
		self.reset_parameters()
		
		

	def forward(self, x):
		h = self.input_shape[1]
		w = self.input_shape[2]
		y = x * self.Weight + self.Bias 

		logdet = h*w*torch.sum(torch.log(torch.abs(self.Weight)))
		return y, logdet

	def inverse(self, y):
		x = (y-self.Bias)/self.Weight
		return x

	def reset_parameters(self):
		nn.init.normal_(self.Weight)
		nn.init.zeros_(self.Bias)

		
class Inv1x1Conv(nn.Module):
	def __init__(self, input_shape, device):
		super(Inv1x1Conv, self).__init__()
		self.input_shape = input_shape
		self.conv = nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1, padding=0, stride=1, bias=False)
		self.device = device
		torch.nn.init.xavier_uniform(self.conv.weight)

	def forward(self, x):
		h = self.input_shape[1]
		w = self.input_shape[2]
		y = self.conv(x)
		#print(self.conv.weight.shape)
		W = self.conv.weight.reshape(self.input_shape[0], self.input_shape[0])
		logdet = h*w*torch.log(torch.abs(torch.det(W)))
		#print(torch.log(torch.abs(torch.det(W))))
		return y, logdet

	def inverse(self, y):
		W = self.conv.weight.reshape(self.input_shape[0], self.input_shape[0])
		y_r = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1])
		W_inv = torch.inverse(W)

		x = torch.matmul(y_r, W_inv)
		x = x.reshape(y.shape[0], y.shape[2], y.shape[3], -1).permute(0, 3, 1, 2)
		
		return x

class ACL(nn.Module):
	def __init__(self, input_shape, hidden_shape, num_splits, num_layers, num_labels):
		super(ACL, self).__init__()
		self.num_splits = num_splits
		self.input_shape = input_shape
		
		self.logs = nn.Sequential(
			nn.Linear(prod(input_shape)+num_labels, hidden_shape),
			nn.ReLU(),
			*[nn.Sequential(
				nn.Linear(hidden_shape, hidden_shape),
				nn.ReLU(),
			) for _ in range(num_layers)],
			nn.Linear(hidden_shape, prod(input_shape)),
			nn.Tanh()
		)
		self.t = nn.Sequential(
			nn.Linear(prod(input_shape)+num_labels, hidden_shape), # This NN takes as input the concatenation of the 2 latent variable groups A and B
			nn.ReLU(),
			*[nn.Sequential(
				nn.Linear(hidden_shape, hidden_shape),
				nn.ReLU(),
			) for _ in range(num_layers)],
			nn.Linear(hidden_shape, prod(input_shape)) # The output however must have the same dimensions as the group C
		)
		if num_splits==3:
			self.logs_2 = nn.Sequential(
				nn.Linear(prod(input_shape)*2+num_labels, hidden_shape),
				nn.ReLU(),
				*[nn.Sequential(
					nn.Linear(hidden_shape, hidden_shape),
					nn.ReLU(),
				) for _ in range(num_layers)],
				nn.Linear(hidden_shape, prod(input_shape)),
				nn.Tanh()
			)
			self.t_2 = nn.Sequential(
				nn.Linear(prod(input_shape)*2+num_labels, hidden_shape), # This NN takes as input the concatenation of the 2 latent variable groups A and B
				nn.ReLU(),
				*[nn.Sequential(
					nn.Linear(hidden_shape, hidden_shape),
					nn.ReLU(),
				) for _ in range(num_layers)],
				nn.Linear(hidden_shape, prod(input_shape)) # The output however must have the same dimensions as the group C
			)

	def forward(self, x, lab, masks):
		x_f = x.flatten(start_dim=1)
		xs = [masks[i]*x_f for i in range(self.num_splits)]
		#logdet=1

		x_combs=[]
		if self.num_splits==3:
			#x_combs=[xs[0]]
			x_12 = torch.cat((xs[1].clone(),xs[0].clone()), dim=1)
			ys = [xs[0].clone()]
			ys.append(masks[1]*(xs[1].clone()*torch.exp(self.logs(torch.cat((xs[0].clone(), lab),dim=1)))+self.t(torch.cat((xs[0].clone(), lab),dim=1))))
			ys.append(masks[2]*(xs[2].clone()*torch.exp(self.logs_2(torch.cat((x_12.clone(), lab),dim=1)))+self.t_2(torch.cat((x_12.clone(), lab),dim=1))))
			#ys.append([masks[i]*(xs[i]*torch.exp(self.logs[i](x_combs[i]))+self.t[i](x_combs[i])) for i in range(1,self.num_splits)]) # WRONG
			logdet = 1 + torch.sum(masks[1]*self.logs(torch.cat((xs[0].clone(), lab),dim=1)), dim=1) + torch.sum(masks[2]*self.logs_2(torch.cat((x_12.clone(), lab),dim=1)), dim=1)

		else:
			#print(masks[1])
			x0 = xs[0].clone()
			x1 = xs[1].clone()
			y0 = x0.clone()
			#import pdb;pdb.set_trace()
			y1 = masks[1]*(x1*torch.exp(self.logs(torch.cat((x0, lab), dim=1)))+self.t(torch.cat((x0, lab), dim=1)))
			ys=[y0.clone(),y1.clone()]
			#print(torch.sum(masks[1]*self.logs[0](x0), dim=1))
			#print(torch.sum(masks[1]*self.logs[0](x0), dim=1).mean())
			logdet = 1. + torch.sum(masks[1]*self.logs(torch.cat((x0, lab), dim=1)), dim=1)

		y=ys[0]
		for i in range(1,len(ys)):
			y+=ys[i]
		y = y.reshape(x.shape)

		return y, logdet

	def inverse(self, y, lab, masks):
		y_f = y.flatten(start_dim=1)
		ys = [masks[i]*y_f for i in range(self.num_splits)]

		if self.num_splits==3:
			#y_combs=[ys[0]]
			y_12 = torch.cat((ys[1],ys[0]), dim=1) 
			xs = [ys[0]]
			xs.append((ys[1]-self.t(torch.cat((ys[0], lab), dim=1))*masks[1])*torch.exp(-self.logs(torch.cat((ys[0], lab), dim=1))))
			xs.append((ys[2]-self.t_2(torch.cat((y_12, lab), dim=1))*masks[2])*torch.exp(-self.logs_2(torch.cat((y_12, lab), dim=1))))

		else:
			y0 = ys[0]
			y1 = ys[1]
			x0 = y0
			x1 = (y1-self.t(torch.cat((y0, lab), dim=1))*masks[1])*torch.exp(-self.logs(torch.cat((y0, lab), dim=1)))
			xs=[x0,x1]

		x=xs[0]
		for i in range(1,len(xs)):
			x+=xs[i]
		x = x.reshape(y.shape)

		return x







class Flow_steps(nn.Module):
	def __init__(self, num_flow_steps, num_layers, num_levels, num_labels, input_shape, hidden_channels, mask, num_splits, device, ch):
		super(Flow_steps, self).__init__()
		self.num_splits = num_splits
		self.num_flow_steps = num_flow_steps
		#self.perm_vec = [torch.randperm(num_splits) for _ in range(num_flow_steps)]
		self.masks=mask

		c = input_shape[0]
		h = input_shape[1]
		w = input_shape[2]
		self.actnorm = Actnorm(input_shape).to(device)
		self.InvConv = Inv1x1Conv(input_shape, device).to(device)
		self.ACL = ACL(input_shape, hidden_channels, num_splits, num_layers, num_labels).to(device)
		

	def forward(self, x, labels, level):
		logdet = 0
		x_out = x
		for k in range(0,self.num_flow_steps):
			x1, logdet1 = self.actnorm(x_out)
			x2, logdet2 = self.InvConv(x1)
			x3, logdet3 = self.ACL(x2, labels, self.masks[k])
			x_out = x3
			logdet += (logdet1+logdet2+logdet3)

		
		return x_out, logdet

	def inverse(self, y, labels, level):
		y_out = y
		for k in range(0,self.num_flow_steps):
			y1 = self.ACL.inverse(y_out, labels, self.masks[self.num_flow_steps-1-k])
			y2 = self.InvConv.inverse(y1)
			y3 = self.actnorm.inverse(y2)
			y_out = y3
		y_out = self.ACL.inverse(y_out, labels, self.masks[self.num_flow_steps-1-k])
		
		return y_out
		

class Glow(nn.Module):
	def __init__(self, input_shape, hidden_shape, n_flow_steps, n_layers, n_levels, n_splits, n_labels, device, chunk_size,
					prior_type='logistic'):
		super().__init__()

		self.n_levels = n_levels
		self.n_splits = n_splits
		self.n_labels = n_labels
		self.output_shapes = [(input_shape[0]*chunk_size**(2*(i+1)), input_shape[1]//(chunk_size**(i+1)), input_shape[2]//(chunk_size**(i+1))) for i in range(n_levels)]
		self.chunk_size = chunk_size
		self.device = device

		masks = [self._get_masks(prod(input_shape), orientation=(i%n_splits)) for i in range(n_flow_steps)]
		self.Fstep = Flow_steps(n_flow_steps, n_layers, n_levels, n_labels, input_shape, hidden_shape, masks, n_splits, device, ch=chunk_size)

		if prior_type == 'logistic':
			self.prior = LogisticDistribution()
		elif prior_type == 'normal':
			self.prior = Normal(0, 1)
		else:
			print("Error: Invalid prior_type")
	
	def forward(self, x, labs):
		z = x
		log_det_jacobian=0
		z_curr = z
		labs = F.one_hot(labs, self.n_labels)
		for lev in range(self.n_levels):
			z, log_det_jacob = self.Fstep(z_curr, labs, lev)
			z_curr=z.clone()
			log_det_jacobian+=log_det_jacob

		
		log_likelihood = torch.sum(self.prior.log_prob(z_curr), dim=(1,2,3)) + log_det_jacobian

		return z_curr, log_likelihood

	def inverse(self, y, labs):
		x = y
		labs = F.one_hot(labs, self.n_labels)
		for lev in range(self.n_levels-1, -1, -1):
			z_curr = x
			x = self.Fstep.inverse(z_curr, labs, lev)
		
		return x


	def _get_masks(self, dim, orientation):
		masks = [torch.zeros(dim) for _ in range(self.n_splits)]
		for i in range(self.n_splits):
			masks[i][i::self.n_splits] = 1.
			masks[i] = masks[i].double().to(self.device)

		perms = list(permutations(range(1,self.n_splits+1)))
		mask_dict = {}
		for i in range(self.n_splits):
			mask_dict.update({i+1:masks[i]})

		masks_ret = [mask_dict[key] for key in perms[orientation%len(perms)]]

		return masks_ret

	
	def sample(self, labels, num_samples):
		y = self.prior.sample([num_samples, *self.input_shape]).view(num_samples, *self.input_shape).type(torch.DoubleTensor).to(self.device)
		generated_samples = self.inverse(y, labels)

		return generated_samples
	
	
