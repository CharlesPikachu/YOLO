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

