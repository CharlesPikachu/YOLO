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
	get targets, for yoloV1.
'''
def build_targets(**kwargs):
	pred_boxes = kwargs.get('pred_boxes')
	target = kwargs.get('target')


'''
Function:
	detectionLayer
	output: (batch_size, nH, nW, nA*5+nC)
	target: (batch_size, (class, x, y, w, h))
'''
class detectionLayer(nn.Module):
	def __init__(self, **kwargs):
		super(detectionLayer, self).__init__()
		self.options = kwargs
	# for forward
	def forward(self, output, target):
		output = output.view(nB, nH, nW, nA*5+nC)
		nB = output.data.size(0)
		nA = self.options.get('num_anchors')
		nC = self.options.get('num_classes')
		nH = output.data.size(1)
		nW = output.data.size(2)
		# x = 
		

