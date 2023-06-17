import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, Distribution

import random
import copy

class LogisticDistribution(Distribution):
	def __init__(self):
		super().__init__()

	def log_prob(self, x):
		return -(F.softplus(x) + F.softplus(-x))

	def sample(self, size):
		z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)

		return torch.log(z) - torch.log(1. - z)

# Not really correct. Make sure to understand the dimension on which the layer is applied
# Also find out about the initialization values

class Actnorm(nn.Module):
	def __init__(self, input_shape):
		super(Actnorm, self).__init__()
		self.input_shape = input_shape
		self.weight = nn.Parameter(torch.Tensor(input_shape[0],1,1))
		self.bias = nn.Parameter(torch.Tensor(input_shape[0],1,1))
		

	def forward(self, x, initialize=False):
		if initialize:
			self.reset_parameters()
		h = self.input_shape[1]
		w = self.input_shape[2]
		y = x * self.weight + self.bias 
		print(self.weight)
		logdet = h*w*torch.sum(torch.log(torch.abs(self.weight)))
		return y, logdet

	def inverse(self, y):
		x = (y-self.bias)/self.weight
		return x

	def reset_parameters(self):
		nn.init.normal_(self.weight)
		nn.init.zeros_(self.bias)

		
class Inv1x1Conv(nn.Module):
	def __init__(self, input_shape):
		super(Inv1x1Conv, self).__init__()
		self.input_shape = input_shape
		self.conv = nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1, padding=0, stride=1, bias=False)
		#torch.nn.init.xavier_uniform(self.conv.weight)

	def forward(self, x):
		h = self.input_shape[1]
		w = self.input_shape[2]
		y = self.conv(x)
		W = self.conv.weight.reshape(self.input_shape[0], self.input_shape[0])
		logdet = h*w*torch.log(torch.abs(torch.det(W)))
		return y, logdet

	def inverse(self, y):
		W = self.conv.weight.reshape(self.input_shape[0], self.input_shape[0])
		x = torch.matmul(torch.inverse(W), y)
		return x

class ACL(nn.Module):
	def __init__(self, input_shape, hidden_channels, num_splits):
		super(ACL, self).__init__()
		self.input_shape = input_shape
		self.num_splits = num_splits
		self.logs = nn.Sequential(
			nn.Conv2d(input_shape[0]//num_splits, hidden_channels, kernel_size=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(hidden_channels, input_shape[0]//num_splits, kernel_size=1),
			nn.Tanh()
		)
		self.t = nn.Sequential(
			nn.Conv2d(input_shape[0]//num_splits, hidden_channels, kernel_size=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(hidden_channels, input_shape[0]//num_splits, kernel_size=1),
			nn.Tanh()
		)

	def forward(self,x, perm):
		x_spls = torch.split(x, x.shape[1]//self.num_splits, dim=1) 
		x_spls = [x_spls[i] for i in perm]
		
		y_i_1 = x_spls[0]
		logdet=1
		ys=[]
		ys.append(y_i_1)
		for i in range(1,self.num_splits):
			y_i = x_spls[i]*torch.exp(self.logs(y_i_1))+self.t(y_i_1)
			ys.append(y_i)
			y_i_1 = y_i
			logdet+=torch.sum(torch.log(torch.abs(torch.exp(self.logs(y_i_1)))))
		
		y = ys[0]
		for i in range(1,self.num_splits):
			y = torch.cat((ys[i],y),dim=1)
		

		return y, logdet

	def inverse(self,y, perm):
		y_spls = torch.split(y, self.num_splits, dim=1)
		perm_new = copy.deepcopy(perm)
		x_spls = [x_spls[i] for i in perm_new.reverse()]
		x_i_1 = y_spls[0]
		xs=[]
		xs.append(x_i_1)
		for i in range(1,self.num_splits):
			x_i = (y_spls[i]-self.t(x_i_1))/torch.exp(self.logs(x_i_1))
			xs.append(x_i)
			x_i_1=x_i

		x = xs[0]
		for i in range(1,self.num_splits):
			x = torch.cat((xs[i],x), dim=1)
		return x





class Flow_steps(nn.Module):
	def __init__(self, num_flow_steps, input_shape, hidden_channels, mask, num_splits):
		super(Flow_steps, self).__init__()
		self.num_splits = num_splits
		self.num_flow_steps = num_flow_steps
		self.perm_vec = [torch.randperm(num_splits) for _ in range(num_flow_steps)]

		self.actnorm = Actnorm(input_shape)
		#self.actnorm2 = Actnorm(input_shape)

		self.InvConv = Inv1x1Conv(input_shape)

		self.ACL = ACL(input_shape, hidden_channels, num_splits)
		

	def forward(self, x, i):
		logdet = 0
		x_out = x
		for k in range(0,self.num_flow_steps):
			if (i==0):
				x1, logdet1 = self.actnorm(x_out, initialize=True)
			else:
				x1, logdet1 = self.actnorm(x_out)
			x2, logdet2 = self.InvConv(x1)
			x3, logdet3 = self.ACL(x2, self.perm_vec[k])
			x_out = x3
			print(logdet1.detach().cpu().numpy(), logdet2.detach().cpu().numpy(), logdet3.detach().cpu().numpy())
			logdet += (logdet1+logdet2+logdet3)
		
		return x_out, logdet

	def inverse(self,y):
		y_out = y
		for k in range(0,self.num_flow_steps):
			y1 = self.ACL.inverse(y_out)
			y2 = self.InvConv.inverse(y1)
			y3 = self.actnorm2.inverse(y2)
			y_out = y3
		
		return y_out
		





class Glow(nn.Module):
	def __init__(self, input_shape, hidden_shape, n_flow_steps, n_levels, n_splits, device,
					prior_type='logistic'):
		super().__init__()

		masks = [self._get_mask(input_shape, orientation=(i % 2 == 0)).to(device) for i in range(n_flow_steps)]
		self.n_levels = n_levels
		self.Fstep = Flow_steps(n_flow_steps, input_shape, hidden_shape, masks, n_splits)

		if prior_type == 'logistic':
			self.prior = LogisticDistribution()
		elif prior_type == 'normal':
			self.prior = Normal(0, 1)
		else:
			print("Error: Invalid prior_type")
	
	def forward(self, x, i):
		z = x
		#z = self.squeeze(x)
		log_det_jacobian=0
		z_discard = []
		z_curr = z
		for lev in range(self.n_levels):
			z, log_det_jacob = self.Fstep(z_curr, i)
			z1, z2 = torch.split(z, z.shape[1]//2, dim=1)
			pd1 = (0,0,0,0,z_curr.shape[1]-z2.shape[1],0,0,0)
			pd2 = (0,0,0,0,0,z_curr.shape[1]-z2.shape[1],0,0)
			z1 = F.pad(z1, pad=pd1, mode='constant', value=0)
			z2 = F.pad(z2, pad=pd2, mode='constant', value=0)
			z_discard.append(z1)
			z_curr = z2
			log_det_jacobian+=log_det_jacob

		'''
		print(z_curr.shape)
		z_curr = self.squeeze(z_curr)
		z, log_det_jacob = self.Fstep(z_curr)
		log_det_jacobian+=log_det_jacob
		z_curr = z
		'''

		return z_curr, z_discard, log_det_jacobian

	def inverse(self, y, z_per_flow):
		z = self.Fstep.inverse(y)
		z = self.unsqueeze(z)
		for lev in range(self.n_levels-1, -1, -1):
			z_lev = z_per_flow[lev]
			z_curr = torch.cat((z_lev, z), dim=1)
			x = self.Fstep.inverse(z_curr)
		x = self.unsqueeze(x)
		return x
			
	
	#def _get_permutation(self, orientation=True):


	def _get_mask(self, dim, orientation=True):
		mask = torch.zeros(dim)
		mask[::2] = 1.
		if orientation:
			mask = 1. - mask # flip mask if orientation is True
		return mask.float()

	def squeeze(self, x, chunk_size=2):
		c = x.shape[1]
		h = x.shape[2]
		w = x.shape[3]
		y = x.permute(0,2,3,1)
		y = y.view(x.shape[0], h//chunk_size, chunk_size, w//chunk_size, chunk_size, c)
		y = y.permute(0,1,3,2,4,5)
		y = y.reshape(x.shape[0], h//chunk_size, w//chunk_size, (chunk_size**2)*c)
		y = y.permute(0,3,1,2)
		return y

	def unsqueeze(self, x, chunk_size=2):
		c_pr = x.shape[1]
		h_pr = x.shape[2]
		w_pr = x.shape[3]

		y = x.permute(0,2,3,1)
		y = y.view(x.shape[0], h_pr*chunk_size, chunk_size, w_pr*chunk_size, chunk_size, c_pr)
		y = y.permute(0,1,3,2,4,5)
		y = y.reshape(x.shape[0], h_pr*chunk_size, w_pr*chunk_size, c//(chunk_size**2))
		y = y.permute(0,3,1,2)
		return y
