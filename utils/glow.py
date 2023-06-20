import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, Distribution

import random
import copy

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
	def __init__(self, input_shape):
		super(Inv1x1Conv, self).__init__()
		self.input_shape = input_shape
		self.conv = nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1, padding=0, stride=1, bias=False)
		torch.nn.init.xavier_uniform(self.conv.weight)

	def forward(self, x):
		h = self.input_shape[1]
		w = self.input_shape[2]
		y = self.conv(x)
		W = self.conv.weight.reshape(self.input_shape[0], self.input_shape[0])
		logdet = h*w*torch.log(torch.abs(torch.det(W)))
		#print(torch.log(torch.abs(torch.det(W))))
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
			logdet+=torch.sum(self.logs(y_i_1), dim=(1,2,3)).mean()
		
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
	def __init__(self, num_flow_steps, num_levels, input_shape, hidden_channels, mask, num_splits, device, ch):
		super(Flow_steps, self).__init__()
		self.num_splits = num_splits
		self.num_flow_steps = num_flow_steps
		self.perm_vec = [torch.randperm(num_splits) for _ in range(num_flow_steps)]

		c = input_shape[0]
		h = input_shape[1]
		w = input_shape[2]

		self.actnorms = nn.ModuleList([Actnorm((c*ch**(2*(i+1)), h//(ch**(i+1)), w//(ch**(i+1)))).to(device) for i in range(num_levels)])
		self.InvConvs = nn.ModuleList([Inv1x1Conv((c*ch**(2*(i+1)), h//(ch**(i+1)), w//(ch**(i+1)))).to(device) for i in range(num_levels)])
		self.ACLs = nn.ModuleList([ACL((c*ch**(2*(i+1)), h//(ch**(i+1)), w//(ch**(i+1))), hidden_channels, num_splits).to(device) for i in range(num_levels)])
		'''
		for i in range(num_levels):
			self.ACLs[i].apply(weights_init)
		'''
		

	def forward(self, x, level):
		logdet = 0
		x_out = x
		for k in range(0,self.num_flow_steps):
			x1, logdet1 = self.actnorms[level](x_out)
			x2, logdet2 = self.InvConvs[level](x1)
			x3, logdet3 = self.ACLs[level](x2, self.perm_vec[k])
			x_out = x3
			#print(logdet1.detach().cpu().numpy(), logdet2.detach().cpu().numpy(), logdet3.detach().cpu().numpy())
			logdet += (logdet1+logdet2+logdet3)

		
		return x_out, logdet

	def inverse(self,y, level):
		y_out = y
		for k in range(0,self.num_flow_steps):
			y1 = self.ACLs[level].inverse(y_out)
			y2 = self.InvConvs[level].inverse(y1)
			y3 = self.actnorms[level].inverse(y2)
			y_out = y3
		
		return y_out
		

class Glow(nn.Module):
	def __init__(self, input_shape, hidden_shape, n_flow_steps, n_levels, n_splits, device, chunk_size,
					prior_type='logistic'):
		super().__init__()

		masks = [self._get_mask(input_shape, orientation=(i % 2 == 0)).to(device) for i in range(n_flow_steps)]
		self.n_levels = n_levels
		self.Fstep = Flow_steps(n_flow_steps, n_levels, input_shape, hidden_shape, masks, n_splits, device, ch=chunk_size)
		self.output_shape = (input_shape[0]*chunk_size**(2*n_levels), input_shape[1]//(chunk_size**n_levels), input_shape[2]//(chunk_size**n_levels))
		self.chunk_size = chunk_size

		if prior_type == 'logistic':
			self.prior = LogisticDistribution()
		elif prior_type == 'normal':
			self.prior = Normal(0, 1)
		else:
			print("Error: Invalid prior_type")
	
	def forward(self, x):
		z = x
		#z = self.squeeze(x)
		log_det_jacobian=0
		z_discard = []
		z_curr = z
		for lev in range(self.n_levels):
			z_curr = self.squeeze(z_curr)
			z, log_det_jacob = self.Fstep(z_curr, lev)
			z1, z2 = torch.split(z, z.shape[1]//2, dim=1)
			pd1 = (0,0,0,0,z_curr.shape[1]-z2.shape[1],0,0,0)
			pd2 = (0,0,0,0,0,z_curr.shape[1]-z2.shape[1],0,0)
			z1 = F.pad(z1, pad=pd1, mode='constant', value=0)
			z2 = F.pad(z2, pad=pd2, mode='constant', value=0)
			z_discard.append(z1)
			z_curr = z2
			log_det_jacobian+=log_det_jacob

		log_likelihood = self.prior.log_prob(z_curr) + log_det_jacobian

		'''
		print(z_curr.shape)
		z_curr = self.squeeze(z_curr)
		z, log_det_jacob = self.Fstep(z_curr)
		log_det_jacobian+=log_det_jacob
		z_curr = z
		'''

		return z_curr, z_discard, log_likelihood

	def inverse(self, y, z_per_flow):
		'''
		z = self.Fstep.inverse(y)
		z = self.unsqueeze(z)
		'''
		x = y
		for lev in range(self.n_levels-1, -1, -1):
			z_lev = z_per_flow[lev]
			z_curr = z_lev+x
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

	def squeeze(self, x):
		c = x.shape[1]
		h = x.shape[2]
		w = x.shape[3]
		y = x.permute(0,2,3,1)
		y = y.view(x.shape[0], h//self.chunk_size, self.chunk_size, w//self.chunk_size, self.chunk_size, c)
		y = y.permute(0,1,3,2,4,5)
		y = y.reshape(x.shape[0], h//self.chunk_size, w//self.chunk_size, (self.chunk_size**2)*c)
		y = y.permute(0,3,1,2)
		return y

	def unsqueeze(self, x):
		c_pr = x.shape[1]
		h_pr = x.shape[2]
		w_pr = x.shape[3]

		y = x.permute(0,2,3,1)
		y = y.view(x.shape[0], h_pr*self.chunk_size, self.chunk_size, w_pr*self.chunk_size, self.chunk_size, c_pr)
		y = y.permute(0,1,3,2,4,5)
		y = y.reshape(x.shape[0], h_pr*self.chunk_size, w_pr*self.chunk_size, c//(self.chunk_size**2))
		y = y.permute(0,3,1,2)
		return y

	
	def sample(self, num_samples):
		y1 = self.prior.sample([num_samples, self.output_shape]).view(num_samples, self.output_shape)
		y2 = [self.prior.sample([num_samples, self.output_shape]).view(num_samples, self.output_shape) for _ in range(self.n_levels)]
		generated_samples = self.inverse(y1, y2)
	
	
