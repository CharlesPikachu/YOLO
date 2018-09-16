'''
Author:
	Charles
Function:
	train for yoloV1-yoloV3
'''
import os
import sys
import math
import config
import torch.optim as optim
from layers import *
from utils.utils import *
from nets.darknet import Darknet
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
		if self.ngpus > 1:
			cur_model = model.module
		else:
			cur_model = model
		train_loader = torch.utils.data.DataLoader(
							myDataset(self.tarinSet,
									  shape=(init_width, init_height),
									  shuffle=True,
									  transform=transforms.Compose([transforms.ToTensor(),]),
									  train=True,
									  seen=cur_model.seen,
									  batch_size=batch_size,
									  num_workers=num_workers),
							batch_size=batch_size,
							shuffle=False,
							**kwargs)
	# initialization
	def __initialization(self):
		self.use_cuda = self.options.get('use_cuda')
		self.backupdir = self.options.get('backupdir')
		self.gpus = self.options.get('gpus')
		self.ngpus = self.options.get('ngpus')
		self.save_interval = self.options.get('save_interval')
		self.tarinSet = self.options.get('tarinSet')
		self.testSet = self.options.get('testSet')
		self.num_workers = self.options.get('num_workers')
		self.weightfile = self.options.get('weightfile')
		self.nsamples = file_lines(self.tarinSet)
		self.batch_size = int(self.net_options.get('batch'))
		self.max_batches = int(self.net_options.get('max_batches'))
		self.learning_rate = float(self.net_options.get('learning_rate'))
		self.steps = [float(step) for step in self.net_options.get('steps').split(',')]
		self.scales = [float(scale) for scale in self.net_options.get('scales').split(',')]
		self.momentum = float(self.net_options.get('momentum'))
		self.decay = float(self.net_options.get('decay'))
		self.max_epochs = math.ceil(self.max_batches * self.batch_size / self.nsamples)
		if not os.path.exists(self.backupdir):
			os.mkdir(self.backupdir)
		if self.use_cuda:
			os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
		self.model = Darknet(self.options)
		if self.options.get('weightfile'):
			self.model.load_weights(weightfile)
			print('[INFO]: %s loaded...' % weightfile)
		self.init_width = self.net_options.get('width')
		self.init_height = self.net_options.get('height')
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
	pass