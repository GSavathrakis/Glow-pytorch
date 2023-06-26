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
	"""
	Description: 
		Activation Normalization Layer. It is used to normalize the data in a linear way, using scale and bias
		parameters that are learnable and thus data-dependent. In the initialization of the class, the scale is 
		initialized with zero mean and unit variance, and the bias with zeros.
	Input:
		Forward pass:
			x: Data Tensor with shape (c x h x w), where c: channels, h:height, w:width
		Inverse pass:
			y: Noise Tensor with shape (c x h x w), where c: channels, h:height, w:width
	Output:
		Forward pass:
			y: Normalized Tensor with shape (c x h x w), where c: channels, h:height, w:width
			logdet: log-determinant of the Actnorm function
		Inverse pass:
			x: Unnormalized Tensor with shape (c x h x w), where c: channels, h:height, w:width
	"""
	def __init__(self, input_shape):
		super(Actnorm, self).__init__()
		self.input_shape = input_shape
		# Initializing trainable parameters
		self.Weight = nn.Parameter(torch.Tensor(input_shape[0],1,1))
		self.Bias = nn.Parameter(torch.Tensor(input_shape[0],1,1))
		self.reset_parameters()
		
		

	def forward(self, x):
		h = self.input_shape[1]
		w = self.input_shape[2]
		y = x * self.Weight + self.Bias # Normalizing the input

		logdet = h*w*torch.sum(torch.log(torch.abs(self.Weight))) # Calculation of the log-determinant
		return y, logdet

	def inverse(self, y):
		x = (y-self.Bias)/self.Weight # Unnormalizing the input
		return x

	def reset_parameters(self):
		nn.init.normal_(self.Weight)
		nn.init.zeros_(self.Bias)

		
class Inv1x1Conv(nn.Module):
	"""
	Description:
		Invertible 1x1 convolution layer. It implements a linear transformation of the input, maintaining its
		original dimensions via the use of a kernel of size 1, and by maintaining the weight matrix, it can be 
		used for the inverse transformation back to the original space.
	Input:
		Forward pass:
			x: Input Tensor from Actnorm layer with shape (c x h x w), where c: channels, h:height, w:width
		Inverse pass:
			y: Input Tensor from ACL layers with shape (c x h x w), where c: channels, h:height, w:width 
	Output:
		Forward pass:
			y: Linearly transformed Tensor with shape (c x h x w), where c: channels, h:height, w:width
			logdet: log-determinant of the Invertible convolution function
		Inverse pass:
			x: Inversely transformed Tensor with shape (c x h x w), where c: channels, h:height, w:width
	"""
	def __init__(self, input_shape, device):
		super(Inv1x1Conv, self).__init__()
		self.input_shape = input_shape
		self.conv = nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1, padding=0, stride=1, bias=False) # Initializing convolutional layer
		self.device = device
		torch.nn.init.xavier_uniform(self.conv.weight)

	def forward(self, x):
		h = self.input_shape[1]
		w = self.input_shape[2]
		y = self.conv(x) # Transformation of the input
		W = self.conv.weight.reshape(self.input_shape[0], self.input_shape[0]) # Transformation weight matrix
		logdet = h*w*torch.log(torch.abs(torch.det(W))) # Calculation of the log determinant
		
		return y, logdet

	def inverse(self, y):
		W = self.conv.weight.reshape(self.input_shape[0], self.input_shape[0]) # Transformation weight matrix
		y_r = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1]) # Permuting the tensor shape as (h x w x c), where c: channels, h:height, w:width
		W_inv = torch.inverse(W) # Calculating the inverse weight matrix

		x = torch.matmul(y_r, W_inv) # Inverse transformation
		x = x.reshape(y.shape[0], y.shape[2], y.shape[3], -1).permute(0, 3, 1, 2) # Bringing tensor back to shape (c x h x w)
		
		return x


class ACL(nn.Module):
	"""
	Description:
		Affine Coupling Layer. It takes the input from the Invertible Convolution layer and flattens it
		along the spatial dimensions. Then it is split to as many parts as the specified number of splits.
		Each part has the same dimensions as the input only in order to isolate each part, masks are used
		to place zeros in the features obscured from each respective split. Subsequently, depending on the 
		number of splits, the first part remains unchanged, and the other parts get transformed as explained 
		in the RealNVP paper (Dinh et al. 2016). The difference here is the we use linear layers instead of convolutional.
	Input:
		Forward pass: 
			x: Input Tensor with shape (c x h x w), where c: channels, h:height, w:width
			masks: List of Tensors with shape (h*w) with 1s for the feature positions that are kept in each split, 0s otherwise
				   List has as many masks as the number of splits
		Inverse pass:
			y: Input Tensor with shape (c x h x w), where c: channels, h:height, w:width
			masks: List of Tensors with shape (h*w) with 1s for the feature positions that are kept in each split, 0s otherwise
				   List has as many masks as the number of splits
	Output:
		Forward pass:
			y: Tensor with shape (c x h x w) which consists of the outputs of the linear layers for each split
			logdet: The log-determinant of the ACL module
		Inverse pass:
			x: Tensor with shape (c x h x w) which consists of the outputs of the inverse linear layers for each split
	"""
	def __init__(self, input_shape, hidden_shape, num_splits, num_levels):
		super(ACL, self).__init__()
		self.num_splits = num_splits
		self.input_shape = input_shape
		
		# Initialization of the scale and translation NNs for possible number of split values 2 and 3
		self.logs = nn.Sequential(
			nn.Linear(prod(input_shape), hidden_shape),
			nn.ReLU(),
			*[nn.Sequential(
				nn.Linear(hidden_shape, hidden_shape),
				nn.ReLU(),
			) for _ in range(num_levels)],
			nn.Linear(hidden_shape, prod(input_shape)),
			nn.Tanh()
		)
		self.t = nn.Sequential(
			nn.Linear(prod(input_shape), hidden_shape), 
			nn.ReLU(),
			*[nn.Sequential(
				nn.Linear(hidden_shape, hidden_shape),
				nn.ReLU(),
			) for _ in range(num_levels)],
			nn.Linear(hidden_shape, prod(input_shape)) 
		)
		if num_splits==3:
			self.logs_2 = nn.Sequential(
				nn.Linear(prod(input_shape)*2, hidden_shape),
				nn.ReLU(),
				*[nn.Sequential(
					nn.Linear(hidden_shape, hidden_shape),
					nn.ReLU(),
				) for _ in range(num_levels)],
				nn.Linear(hidden_shape, prod(input_shape)),
				nn.Tanh()
			)
			self.t_2 = nn.Sequential(
				nn.Linear(prod(input_shape)*2, hidden_shape), 
				nn.ReLU(),
				*[nn.Sequential(
					nn.Linear(hidden_shape, hidden_shape),
					nn.ReLU(),
				) for _ in range(num_levels)],
				nn.Linear(hidden_shape, prod(input_shape)) 
			)

	def forward(self, x, masks):
		x_f = x.flatten(start_dim=1) # Flattening the spatial dimensions
		xs = [masks[i]*x_f for i in range(self.num_splits)] # Splitting into num_splits parts

		x_combs=[]
		if self.num_splits==3:
			# If there are 3 parts, the output of the 3rd part uses the concatenated input of the 1st and 2nd parts as input to the NNs
			x_12 = torch.cat((xs[1].clone(),xs[0].clone()), dim=1)
			# Calculation of the output
			ys = [xs[0].clone()]
			ys.append(masks[1]*(xs[1].clone()*torch.exp(self.logs(xs[0].clone()))+self.t(xs[0].clone())))
			ys.append(masks[2]*(xs[2].clone()*torch.exp(self.logs_2(x_12))+self.t_2(x_12)))
			logdet = 1 + torch.sum(masks[1]*self.logs(xs[0].clone()), dim=1) + torch.sum(masks[2]*self.logs_2(x_12), dim=1) # Calculation of the log-determinant

		else:
			x0 = xs[0].clone()
			x1 = xs[1].clone()
			# Calculation of the output
			y0 = x0.clone()
			y1 = masks[1]*(x1*torch.exp(self.logs(x0))+self.t(x0))
			ys=[y0.clone(),y1.clone()]
			logdet = 1. + torch.sum(masks[1]*self.logs(x0), dim=1) # Calculation of the log-determinant

		# Concatenating the parts
		y=ys[0]
		for i in range(1,len(ys)):
			y+=ys[i]
		y = y.reshape(x.shape)

		return y, logdet

	def inverse(self, y, masks):
		y_f = y.flatten(start_dim=1) # Flattening the spatial dimensions
		ys = [masks[i]*y_f for i in range(self.num_splits)] # Splitting into num_splits parts

		# Similar method as in forward pass
		if self.num_splits==3:
			y_12 = torch.cat((ys[1],ys[0]), dim=1) 
			xs = [ys[0]]
			xs.append((ys[1]-self.t(ys[0])*masks[1])*torch.exp(-self.logs(ys[0])))
			xs.append((ys[2]-self.t_2(y_12)*masks[2])*torch.exp(-self.logs_2(y_12)))

		else:
			y0 = ys[0]
			y1 = ys[1]
			x0 = y0
			x1 = (y1-self.t(y0)*masks[1])*torch.exp(-self.logs(y0))
			xs=[x0,x1]

		x=xs[0]
		for i in range(1,len(xs)):
			x+=xs[i]
		x = x.reshape(y.shape)

		return x



class Flow_steps(nn.Module):
	"""
	Description:
		Module build for flow steps. It consists of the sequence Actnorm, Invertible 1x1 convolution and ACL.
	Input:
		Forward pass:
			x: Input Tensor with shape (c x h x w), where c: channels, h:height, w:width
			level: Current training level (see Glow description)
		Inverse pass:
			y: Input Tensor with shape (c x h x w), where c: channels, h:height, w:width
			level: Current training level (see Glow description)
	Output:
		Forward pass:
			x_out: Tensor with shape (c x h x w), where c: channels, h:height, w:width. The output of final ACL
			logdet: The log-determinant of one flow step. It is the sum of the log-determinants of each component
		Inverse pass:
			y_out: Tensor with shape (c x h x w), where c: channels, h:height, w:width. The output of the inverse flow 
				   following the sequence ACL --> Invertible Convolution --> Actnorm
	"""
	def __init__(self, num_flow_steps, num_levels, num_layers, input_shape, hidden_channels, mask, num_splits, device, ch):
		super(Flow_steps, self).__init__()
		self.num_splits = num_splits
		self.num_flow_steps = num_flow_steps
		self.masks=mask

		c = input_shape[0]
		h = input_shape[1]
		w = input_shape[2]
		#Initializing flow components
		self.actnorm = Actnorm(input_shape).to(device)
		self.InvConv = Inv1x1Conv(input_shape, device).to(device)
		self.ACL = ACL(input_shape, hidden_channels, num_splits, num_layers).to(device)
		

	def forward(self, x, level):
		logdet = 0
		x_out = x
		# For each flow step we calculate the output from the components' sequence and re give as input to the next flow step
		for k in range(0,self.num_flow_steps):
			x1, logdet1 = self.actnorm(x_out)
			x2, logdet2 = self.InvConv(x1)
			x3, logdet3 = self.ACL(x2, self.masks[k])
			x_out = x3
			logdet += (logdet1+logdet2+logdet3) # Calculation of the log-determinant of each flow step

		
		return x_out, logdet

	def inverse(self,y, level):
		y_out = y
		# Similar to forward pass but the sequence is inversed
		for k in range(0,self.num_flow_steps):
			y1 = self.ACL.inverse(y_out, self.masks[self.num_flow_steps-1-k])
			y2 = self.InvConv.inverse(y1)
			y3 = self.actnorm.inverse(y2)
			y_out = y3
		
		return y_out
		

class Glow(nn.Module):
	"""
	Description:
		The Glow model. The samples get through the forward pass as many times as the number of training levels. In each level, they pass
		through as many flow steps as defined in the model's initialization. In each flow step, the ordering of the masks used for the split,
		changes according to a permutation strategy in order to ensure that the model is generalizable. The log-likelihood is the sum of the 
		log probability of the prior distribution of the data with the log-determinant of the jacobian.
	Input:
		Forward pass:
			x: Input Tensors with shape (N x c x h x w) where, N: batch size, c: channels, h: height, w: width
		Inverse pass:
			y: Noise Tensors with shape (N x c x h x w) where, N: batch size, c: channels, h: height, w: width
	Output:
		Forward pass:
			z_curr: Final output of the model after the specified number of levels. Tensor with shape (N x c x h x w) where, N: batch size, c: channels, h: height, w: width
			log_likelihood: The log-likelihood calculated after the Glow orward pass
	"""
	def __init__(self, input_shape, hidden_shape, n_flow_steps, n_layers, n_levels, n_splits, device, chunk_size,
					prior_type='logistic'):
		super().__init__()

		self.n_levels = n_levels
		self.n_splits = n_splits

		self.input_shape = input_shape
		self.chunk_size = chunk_size
		self.device = device

		masks = [self._get_masks(prod(input_shape), orientation=(i%n_splits)) for i in range(n_flow_steps)] # Mask creation. Each with different combination of 0s and 1s according to the flow step
		self.Fstep = Flow_steps(n_flow_steps, n_levels, n_layers, input_shape, hidden_shape, masks, n_splits, device, ch=chunk_size) # Initializing Flow steps

		# Initializing prior distribution
		if prior_type == 'logistic':
			self.prior = LogisticDistribution()
		elif prior_type == 'normal':
			self.prior = Normal(0, 1)
		else:
			print("Error: Invalid prior_type")
	
	def forward(self, x):
		z = x
		log_det_jacobian=0
		z_curr = z
		# Training samples for n levels
		for lev in range(self.n_levels):
			z, log_det_jacob = self.Fstep(z_curr, lev)
			z_curr=z.clone()
			log_det_jacobian+=log_det_jacob

		
		log_likelihood = torch.sum(self.prior.log_prob(z_curr), dim=(1,2,3)) + log_det_jacobian # Calculation of the log-likelihood

		return z_curr, log_likelihood

	def inverse(self, y):
		#Inverse pass back to data space
		x = y
		for lev in range(self.n_levels-1, -1, -1):
			z_curr = x
			x = self.Fstep.inverse(z_curr, lev)
		
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

	
	def sample(self, num_samples):
		y = self.prior.sample([num_samples, *self.input_shape]).view(num_samples, *self.input_shape).type(torch.DoubleTensor).to(self.device)
		
		generated_samples = self.inverse(y)

		return generated_samples
	
	
