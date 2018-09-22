'''
Author: 
	Charles
Function:
	Create yolo network(https://github.com/pjreddie/darknet) with pytorch0.3.1.
'''
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../')
from layers import *
from utils.utils import *


'''
Function:
	MaxPool2d, stride = 1
'''
class MaxPoolStride1(nn.Module):
	def __init__(self):
		super(MaxPoolStride1, self).__init__()
	def forward(self, x):
		x_pad = F.pad(x, (0, 1, 0, 1), mode='replicate')
		x = F.max_pool2d(x_pad, 2, stride=1)
		return x


'''
Function:
	Upsample, expand H, W using their own elements.
	(batch_size, in_channels, h, w) -> (batch_size, in_channels, h*stride, w*stride)
'''
class Upsample(nn.Module):
	def __init__(self, stride=2):
		super(Upsample, self).__init__()
		self.stride = stride
	def forward(self, x):
		assert(x.data.dim() == 4)
		B = x.data.size(0)
		C = x.data.size(1)
		H = x.data.size(2)
		W = x.data.size(3)
		x = x.view(B, C, H, 1, W, 1).expand(B, C, H, self.stride, W, self.stride).contiguous().view(B, C, H*self.stride, W*self.stride)
		return x


'''
Function:
	Reorg, (B, C, H, W) -> (B, hs*ws*C, H//hs, W//ws)
'''
class Reorg(nn.Module):
	def __init__(self, stride=2):
		super(Reorg, self).__init__()
		self.stride = stride
	def forward(self, x):
		assert(x.data.dim() == 4)
		B = x.data.size(0)
		C = x.data.size(1)
		H = x.data.size(2)
		W = x.data.size(3)
		assert(H % self.stride == 0)
		assert(W % self.stride == 0)
		w_stride = self.stride
		h_stride = self.stride
		x = x.view(B, C, H//h_stride, h_stride, W//w_stride, w_stride).transpose(3, 4).contiguous()
		x = x.view(B, C, (H//h_stride)*(W//w_stride), h_stride*w_stride).transpose(2, 3).contiguous()
		x = x.view(B, C, h_stride*w_stride, H//h_stride, W//w_stride).transpose(1, 2).contiguous()
		x = x.view(B, h_stride*w_stride*C, H//h_stride, W//w_stride)
		return x


'''
Function:
	Global AvgPool2d.
'''
class GlobalAvgPool2d(nn.Module):
	def __init__(self):
		super(GlobalAvgPool2d, self).__init__()
	def forward(self, x):
		N = x.data.size(0)
		C = x.data.size(1)
		H = x.data.size(2)
		W = x.data.size(3)
		x = F.avg_pool2d(x, (H, W))
		x = x.view(N, C)
		return x


'''
Function:
	empty module.
'''
class EmptyModule(nn.Module):
	def __init__(self):
		super(EmptyModule, self).__init__()
	def forward(self, x):
		return x


'''
Function:
	Create yolo network.
'''
class Darknet(nn.Module):
	def __init__(self, options):
		super(Darknet, self).__init__()
		self.blocks = CfgParser().parser(options.get('cfgfile'), is_print=True)
		# record some information of the model.
		self.header = torch.IntTensor([0, 0, 0, 0, 0])
		self.seen = self.header[3]
		self.det_strides = []
		self.options = options
		self.models = self.create_network()
	# net forward
	def forward(self, x, target=None):
		self.seen += x.data.size(0)
		ind = -2
		loss = 0
		res = []
		outputs = dict()
		for block in self.blocks:
			ind += 1
			if block['layer_type'] == 'net':
				continue
			elif block['layer_type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'local', 'dropout']:
				x = self.models[ind](x)
				outputs[ind] = x
			elif block['layer_type'] == 'connected':
				batch_size = x.size(0)
				height = x.size(2)
				width = x.size(3)
				x = x.transpose(1, 3).transpose(1, 2).contiguous()
				x = x.view(x.size(0), -1)
				x = self.models[ind](x)
				# x = x.view((batch_size, height, width, -1))
			elif block['layer_type'] == 'route':
				layers = block['layers'].split(',')
				layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
				if len(layers) == 1:
					x = outputs[layers[0]]
					outputs[ind] = x
				elif len(layers) == 2:
					x1 = outputs[layers[0]]
					x2 = outputs[layers[1]]
					x = torch.cat((x1, x2), 1)
					outputs[ind] = x
			elif block['layer_type'] == 'shortcut':
				from_layer = int(block['from'])
				activation = block['activation']
				from_layer = from_layer if from_layer > 0 else from_layer + ind
				x1 = outputs[from_layer]
				x2 = outputs[ind-1]
				x  = x1 + x2
				if activation == 'leaky':
					x = F.leaky_relu(x, 0.1, inplace=True)
				elif activation == 'relu':
					x = F.relu(x, inplace=True)
				outputs[ind] = x
			# yoloV1, yoloV2, yoloV3
			elif block['layer_type'] in ['detection', 'region', 'yolo']:
				if self.options.get('mode') == 'train':
					self.models[ind].seen = self.seen
					loss += self.models[ind](x, target)
				else:
					res.append(x)
			# for resnet, too lazy to realize, o(╥﹏╥)o
			elif block['layer_type'] == 'cost':
				continue
			else:
				print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
				sys.exit(0)
		if self.options.get('mode') == 'train':
			return loss
		else:
			return x if len(res) < 2 else res
	# create netword
	def create_network(self):
		models = nn.ModuleList()
		conv_id = 0
		out_filters = []
		out_strides = []
		prev_stride = 1
		for block in self.blocks:
			if block['layer_type'] == 'net':
				init_width = int(block['width'])
				init_height = int(block['height'])
				prev_filters = int(block['channels'])
				continue
			elif block['layer_type'] in ['convolutional', 'local']:
				conv_id = conv_id + 1
				try:
					batch_normalize = int(block['batch_normalize'])
				except:
					batch_normalize = False
				filters = int(block['filters'])
				kernel_size = int(block['size'])
				stride = int(block['stride'])
				is_pad = int(block['pad'])
				pad = (kernel_size-1)//2 if is_pad else 0
				activation = block['activation']
				model = nn.Sequential()
				if batch_normalize:
					model.add_module('conv{}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
					model.add_module('bn{}'.format(conv_id), nn.BatchNorm2d(filters))
				else:
					model.add_module('conv{}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
				if activation == 'leaky':
					model.add_module('leaky{}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
				elif activation == 'relu':
					model.add_module('relu{}'.format(conv_id), nn.ReLU(inplace=True))
				prev_filters = filters
				out_filters.append(prev_filters)
				prev_stride = stride * prev_stride
				out_strides.append(prev_stride)
				models.append(model)
			elif block['layer_type'] == 'maxpool':
				pool_size = int(block['size'])
				stride = int(block['stride'])
				if stride > 1:
					model = nn.MaxPool2d(pool_size, stride)
				else:
					model = MaxPoolStride1()
				out_filters.append(prev_filters)
				prev_stride = stride * prev_stride
				out_strides.append(prev_stride)
				models.append(model)
			elif block['layer_type'] == 'avgpool':
				model = GlobalAvgPool2d()
				out_filters.append(prev_filters)
				models.append(model)
			elif block['layer_type'] == 'softmax':
				model = nn.Softmax()
				out_strides.append(prev_stride)
				out_filters.append(prev_filters)
				models.append(model)
			elif block['layer_type'] == 'cost':
				if block['type'] == 'sse':
					model = nn.MSELoss(size_average=True)
				elif block['type'] == 'L1':
					model = nn.L1Loss(size_average=True)
				elif block['type'] == 'smooth':
					model = nn.SmoothL1Loss(size_average=True)
				out_filters.append(1)
				out_strides.append(prev_stride)
				models.append(model)
			elif block['layer_type'] == 'reorg':
				stride = int(block['stride'])
				prev_filters = stride * stride * prev_filters
				out_filters.append(prev_filters)
				prev_stride = prev_stride * stride
				out_strides.append(prev_stride)
				models.append(Reorg(stride))
			elif block['layer_type'] == 'upsample':
				stride = int(block['stride'])
				out_filters.append(prev_filters)
				prev_stride = prev_stride // stride
				out_strides.append(prev_stride)
				models.append(Upsample(stride))
			elif block['layer_type'] == 'route':
				layers = block['layers'].split(',')
				ind = len(models)
				layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
				if len(layers) == 1:
					prev_filters = out_filters[layers[0]]
					prev_stride = out_strides[layers[0]]
				elif len(layers) == 2:
					assert(layers[0] == ind - 1)
					prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
					prev_stride = out_strides[layers[0]]
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				models.append(EmptyModule())
			elif block['layer_type'] == 'shortcut':
				ind = len(models)
				prev_filters = out_filters[ind-1]
				prev_stride = out_strides[ind-1]
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				models.append(EmptyModule())
			elif block['layer_type'] == 'connected':
				filters = int(block['output'])
				stride = out_strides[-1]
				if block['activation'] == 'linear':
					model = nn.Linear(prev_filters * (init_height//stride) * (init_width//stride), filters)
				elif block['activation'] == 'leaky':
					model = nn.Sequential(
								nn.Linear(prev_filters, filters),
								nn.LeakyReLU(0.1, inplace=True))
				elif block['activation'] == 'relu':
					model = nn.Sequential(
								nn.Linear(prev_filters, filters),
								nn.ReLU(inplace=True))
				prev_filters = filters
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				models.append(model)
			elif block['layer_type'] == 'dropout':
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				prob = float(block['probability'])
				model = nn.Dropout(p=prob)
				models.append(model)
			elif block['layer_type'] == 'detection':
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				num_boxes = int(block['num'])
				num_classes = int(block['classes'])
				object_scale = float(block['object_scale'])
				noobject_scale = float(block['noobject_scale'])
				class_scale = float(block['class_scale'])
				coord_scale = float(block['coord_scale'])
				deLayer = detectionLayer.detectionLayer(num_classes=num_classes,
														coord_scale=coord_scale,
														class_scale=class_scale,
														noobject_scale=noobject_scale,
														object_scale=object_scale)
				models.append(deLayer)
				self.det_strides.append(prev_stride)
			elif block['layer_type'] == 'region':
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)
				num_anchors = int(block['num'])
				num_classes = int(block['classes'])
				anchors = [float(i) for i in block['anchors'].split(',')]
				noobject_scale = float(block['noobject_scale'])
				object_scale = float(block['object_scale'])
				sil_thresh = float(block['thresh'])
				seen = self.seen
				max_object = self.options.get('max_object')
				by_stride = self.options.get('by_stride')
				coord_scale = float(block['coord_scale'])
				class_scale = float(block['class_scale'])
				reLayer = regionLayer.regionLayer(num_anchors=num_anchors,
												  num_classes=num_classes,
												  stride=prev_stride,
												  anchors=anchors,
												  noobject_scale=noobject_scale,
												  object_scale=object_scale,
												  sil_thresh=sil_thresh,
												  seen=seen,
												  max_object=max_object,
												  by_stride=by_stride,
												  coord_scale=coord_scale,
												  class_scale=class_scale)
				self.det_strides.append(prev_stride)
				models.append(reLayer)
			elif block['layer_type'] == 'yolo':
				out_filters.append(prev_filters)
				out_strides.append(prev_stride)

				yoLayer = yoloLayer.yoloLayer()
				models.append(yoLayer)
				self.det_strides.append(prev_stride)
			else:
				print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
				sys.exit(0)
		return models
	# load weights
	def load_weights(self, weightfile):
		with open(weightfile, 'rb') as fp:
			# before yolo3, weights get from https://github.com/pjreddie/darknet count = 4.
			header = np.fromfile(fp, count=5, dtype=np.int32)
			self.header = torch.from_numpy(header)
			self.seen = self.header[3]
			buf = np.fromfile(fp, dtype=np.float32)
		start = 0
		ind = -2
		for block in self.blocks:
			if start >= buf.size:
				break
			ind = ind + 1
			if block['layer_type'] in ['net', 'maxpool', 'reorg',
									   'upsample', 'route', 'shortcut',
									   'region', 'yolo', 'avgpool',
									   'softmax', 'cost', 'detection',
									   'dropout']:
				continue
			elif block['layer_type'] in ['convolutional', 'local']:
				model = self.models[ind]
				try:
					batch_normalize = int(block['batch_normalize'])
				except:
					batch_normalize = False
				if batch_normalize:
					start = load_conv_bn(buf, start, model[0], model[1])
				else:
					start = load_conv(buf, start, model[0])
			elif block['layer_type'] == 'connected':
				model = self.models[ind]
				if block['activation'] != 'linear':
					start = load_fc(buf, start, model[0])
				else:
					start = load_fc(buf, start, model)
			else:
				print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
				sys.exit(0)
	# save weights
	def save_weights(self, outfile):
		fp = open(outfile, 'wb')
		self.header[3] = self.seen
		header = self.header
		header.numpy().tofile(fp)
		ind = -1
		for blockId in range(1, len(self.blocks)):
			ind = ind + 1
			block = self.blocks[blockId]
			if block['layer_type'] in ['convolutional', 'local']:
				model = self.models[ind]
				try:
					batch_normalize = int(block['batch_normalize'])
				except:
					batch_normalize = False
				if batch_normalize:
					save_conv_bn(fp, model[0], model[1])
				else:
					save_conv(fp, model[0])
			elif block['type'] == 'connected':
				model = self.models[ind]
				if block['activation'] != 'linear':
					save_fc(fc, model[0])
				else:
					save_fc(fc, model)
			elif block['layer_type'] in ['net', 'maxpool', 'reorg',
										 'upsample', 'route', 'shortcut',
										 'region', 'yolo', 'avgpool',
										 'softmax', 'cost', 'detection',
										 'dropout']:
				continue
			else:
				print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
				sys.exit(0)
		fp.close()