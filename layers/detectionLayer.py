'''
Author:
	Charles
Function:
	detectionLayer, compute loss for yoloV1
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


'''
Function:
	detectionLayer
	output: (batch_size, num_anchors, (num_classes+5), nH, nW)
	target: (class, x, y, w, h)
'''
class detectionLayer(nn.Module):
	def __init__(self, **kwargs):
		super(detectionLayer, self).__init__()
		self.options = kwargs
	# for forward
	def forward(self, output, target):
		coord_mask = target[:, :, :, ]
