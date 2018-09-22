'''
Author:
	Charles
Function:
	yoloLayer, compute loss for yoloV3
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable





'''
Function:
	yoloLayer, for yoloV3
	output: (batch_size, num_anchors, (num_classes+5), nH, nW)
	target: (batch_size, (class, x, y, w, h))
'''
class yoloLayer(nn.Module):
	def __init__(self, **kwargs):
		super(yoloLayer, self).__init__()
		self.options = kwargs
		self.seen = self.options.get('seen')