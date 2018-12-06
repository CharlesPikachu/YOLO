'''
Author:
	Charles
Function:
	train for yoloV1-yoloV3
'''
import os
import sys
import math
import time
import config
import torch.nn as nn
import torch.optim as optim
from layers import *
from utils.utils import *
from nets.darknet import Darknet
from evalpg.eval import evalModel
from torchvision import transforms
from torch.autograd import Variable
from dataset.dataset import myDataset


'''
Function:
	class for training
Input:
	yolo_type: 'yolo1' or 'yolo2' or 'yolo3'.
'''
# ----------------------------------------------------------------------------------------------------------------------------------
class train():
	def __init__(self, yolo_type='yolo2'):
		self.yolo_type = yolo_type
		if yolo_type == 'yolo1':
			self.options = config.yolo1_options
		elif yolo_type == 'yolo2':
			self.options = config.yolo2_options
		elif yolo_type == 'yolo3':
			self.options = config.yolo3_options
		else:
			raise ValueError('Unknow yolo_type <%s>...' % yolo_type)
		cfgparser = CfgParser()
		self.net_options = cfgparser.parser(cfgfile=self.options.get('cfgfile'))[0]
		self.__initialization()
	# start to train.
	def start(self):
		for epoch in range(self.init_epoch, self.max_epochs):
			self.__train_epoch(epoch)
	# train for one epoch.
	def __train_epoch(self, epoch):
		if self.use_cuda:
			if self.ngpus > 1:
				cur_model = self.model.module
			else:
				cur_model = self.model
		else:
			cur_model = self.model
		train_loader = torch.utils.data.DataLoader(
							myDataset(root=[self.trainSet, self.trainlabpth],
									  shape=(self.init_width, self.init_height),
									  shuffle=True,
									  transform=transforms.Compose([transforms.ToTensor(),]),
									  is_train=True,
									  seen=cur_model.seen,
									  batch_size=self.batch_size,
									  num_workers=self.num_workers,
									  is_multiscale=self.is_multiscale,
									  jitter=self.jitter,
									  hue=self.hue,
									  saturation=self.saturation,
									  exposure=self.exposure,
									  max_object=self.max_object),
							batch_size=self.batch_size,
							shuffle=False,
							**self.kwargs)
		logging('epoch %d, processed %d samples, processed_batches %d' % (epoch, epoch * len(train_loader.dataset), self.processed_batches))
		self.model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			lr = self.__adjust_lr(self.optimizer, self.processed_batches)
			self.processed_batches += 1
			if self.use_cuda:
				data = data.cuda()
			data, target = Variable(data), Variable(target)
			self.optimizer.zero_grad()
			cur_model.seen += data.data.size(0)
			loss = self.model(data, target)
			if self.ngpus > 1:
				loss = loss.sum()
			loss.backward()
			self.optimizer.step()
		if ((epoch + 1) % self.save_interval == 0) or ((epoch + 1) == self.max_epochs):
			logging('save weights to %s/%06d.weights' % (self.backupdir, epoch+1))
			cur_model.seen = (epoch + 1) * len(train_loader.dataset)
			cur_model.save_weights('%s/%06d.weights' % (self.backupdir, epoch+1))
			self.EM.eval(self.model)
	# initialization
	def __initialization(self):
		self.use_cuda = self.options.get('use_cuda')
		self.backupdir = self.options.get('backupdir')
		self.gpus = self.options.get('gpus')
		self.ngpus = self.options.get('ngpus')
		self.save_interval = self.options.get('save_interval')
		self.trainSet = self.options.get('trainSet')
		self.testSet = self.options.get('testSet')
		self.trainlabpth = self.options.get('trainlabpth')
		self.testlabpth = self.options.get('testlabpth')
		self.num_workers = self.options.get('num_workers')
		self.is_multiscale = self.options.get('is_multiscale')
		self.nsamples = file_lines(self.trainSet)
		self.batch_size = int(self.net_options.get('batch'))
		self.max_batches = int(self.net_options.get('max_batches'))
		self.max_object = self.options.get('max_object')
		self.learning_rate = float(self.net_options.get('learning_rate'))
		self.steps = [float(step) for step in self.net_options.get('steps').split(',')]
		self.scales = [float(scale) for scale in self.net_options.get('scales').split(',')]
		self.momentum = float(self.net_options.get('momentum'))
		self.decay = float(self.net_options.get('decay'))
		self.jitter = self.options.get('jitter')
		self.saturation = float(self.net_options.get('saturation'))
		self.exposure = float(self.net_options.get('exposure'))
		self.hue = float(self.net_options.get('hue'))
		self.max_epochs = math.ceil(self.max_batches * self.batch_size / self.nsamples)
		if not os.path.exists(self.backupdir):
			os.mkdir(self.backupdir)
		if self.use_cuda:
			os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
		self.model = Darknet(self.options)
		weightfile = self.options.get('weightfile')
		if weightfile:
			self.model.load_weights(weightfile)
			print('[INFO]: %s loaded...' % weightfile)
		self.init_width = int(self.net_options.get('width'))
		self.init_height = int(self.net_options.get('height'))
		self.init_epoch = self.model.seen // self.nsamples
		self.processed_batches = self.model.seen // self.nsamples
		self.kwargs = {'num_workers': self.num_workers, 'pin_memory': True} if self.use_cuda else {}
		if self.use_cuda:
			if self.ngpus > 1:
				self.model = nn.DataParallel(self.model).cuda()
			else:
				self.model = self.model.cuda()
		self.optimizer = optim.SGD(self.model.parameters(), 
								   lr=self.learning_rate/self.batch_size, 
								   momentum=self.momentum, 
								   dampening=0, 
								   weight_decay=self.decay*self.batch_size)
		self.EM = evalModel(num_workers=self.num_workers,
							use_cuda=self.use_cuda,
							testSet=self.testSet,
							testlabpth=self.testlabpth,
							init_width=self.init_width,
							init_height=self.init_height,
							batch_size=self.batch_size,
							ngpus=self.ngpus,
							by_stride=self.options.get('by_stride'),
							conf_thresh=self.options.get('conf_thresh'),
							iou_thresh=self.options.get('iou_thresh'),
							max_object=self.max_object,
							is_multiscale=self.is_multiscale,
							yolo_type=self.yolo_type,
							nms_thresh=self.options.get('nms_thresh'))
	# adjust learning rate.
	def __adjust_lr(self, optimizer, batch_idx):
		lr = self.learning_rate
		for i in range(len(self.steps)):
			scale = self.scales[i] if i < len(self.scales) else 1
			if batch_idx >= self.steps[i]:
				lr = lr * scale
				if batch_idx == self.steps[i]:
					break
			else:
				break
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr/self.batch_size
		return lr
# ----------------------------------------------------------------------------------------------------------------------------------





if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--version', help='Choose the version of yolo.')
	args = parser.parse_args()
	t = train(yolo_type=args.version)
	t.start()