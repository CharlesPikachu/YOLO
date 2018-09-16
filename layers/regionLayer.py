'''
Author:
	Charles
Function:
	regionLayer, compute loss for yoloV2.
'''
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.append('..')
from utils.utils import *


'''
Function:
	get targets, for yoloV2
'''
def build_targets(**kwargs):
	# parse parameters.
	pred_boxes = kwargs.get('pred_boxes')
	target = kwargs.get('target')
	anchors = kwargs.get('anchors')
	nB = target.size(0)
	nA = kwargs.get('num_anchors')
	nC = kwargs.get('num_classes')
	nW = kwargs.get('nW')
	nH = kwargs.get('nH')
	noobject_scale = kwargs.get('noobject_scale')
	object_scale = kwargs.get('object_scale')
	sil_thresh = kwargs.get('sil_thresh')
	seen = kwargs.get('seen')
	max_object = kwargs.get('max_object')
	anchor_step = len(anchors) // nA
	conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
	coord_mask = torch.zeros(nB, nA, nH, nW)
	cls_mask = torch.zeros(nB, nA, nH, nW)
	tx = torch.zeros(nB, nA, nH, nW)
	ty = torch.zeros(nB, nA, nH, nW)
	tw = torch.zeros(nB, nA, nH, nW)
	th = torch.zeros(nB, nA, nH, nW)
	tconf = torch.zeros(nB, nA, nH, nW)
	tcls = torch.zeros(nB, nA, nH, nW)
	nAnchors = nA * nH * nW
	nPixels  = nH * nW
	# for each ground truth, find iou > sil_thresh.
	for b in range(nB):
		cur_pred_boxes = pred_boxes[b*nAnchors: (b+1)*nAnchors].t()
		cur_ious = torch.zeros(nAnchors)
		for t in range(max_object):
			if target[b][t*5+1] == 0:
				break
			gx = target[b][t*5+1] * nW
			gy = target[b][t*5+2] * nH
			gw = target[b][t*5+3] * nW
			gh = target[b][t*5+4] * nH
			cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
			cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
		conf_mask[b][cur_ious>sil_thresh] = 0
	if seen < 12800:
		tx.fill_(0.5)
		ty.fill_(0.5)
		tw.zero_()
		th.zero_()
		coord_mask.fill_(1)
	# the number of ground truth.
	nGT = 0
	nCorrect = 0
	for b in range(nB):
		for t in range(max_object):
			if target[b][t*5+1] == 0:
				break
			nGT = nGT + 1
			best_iou = 0.0
			best_n = -1
			gx = target[b][t*5+1] * nW
			gy = target[b][t*5+2] * nH
			gi = int(gx)
			gj = int(gy)
			gw = target[b][t*5+3] * nW
			gh = target[b][t*5+4] * nH
			# find the best anchor to match ground truth.
			gt_box = [0, 0, gw, gh]
			for n in range(nA):
				aw = anchors[anchor_step*n]
				ah = anchors[anchor_step*n+1]
				anchor_box = [0, 0, aw, ah]
				iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
				if iou > best_iou:
					best_iou = iou
					best_n = n
			gt_box = [gx, gy, gw, gh]
			pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
			coord_mask[b][best_n][gj][gi] = 1
			cls_mask[b][best_n][gj][gi] = 1
			conf_mask[b][best_n][gj][gi] = object_scale
			tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
			ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
			tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
			th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
			iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
			tconf[b][best_n][gj][gi] = iou
			tcls[b][best_n][gj][gi] = target[b][t*5]
			# iou > 0.5 as the correct one.
			if iou > 0.5:
				nCorrect += 1
	return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


'''
Function:
	regionLayer, for yoloV2
	output: (batch_size, num_anchors, (num_classes+5), nH, nW)
	target: (class, x, y, w, h)
'''
class regionLayer(nn.Module):
	def __init__(self, **kwargs):
		super(regionLayer, self).__init__()
		self.options = kwargs
	# for forward
	def forward(self, output, target):
		'''
		Part1: Parse output
		'''
		# size of batch
		nB = output.data.size(0)
		# number of anchors per grid
		nA = self.options.get('num_anchors')
		# number of classes
		nC = self.options.get('num_classes')
		# number of grids is 'H * N'
		nH = output.data.size(2)
		nW = output.data.size(3)
		anchors = [anchor/self.options.get('stride') for anchor in self.options.get('anchors')]
		# 5 -> (x, y, w, h) and box confidence.
		output = output.view(nB, nA, (5+nC), nH, nW)
		x = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
		y = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
		w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
		h = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
		conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
		cls_ = output.index_select(2, Variable(torch.linspace(5, 5+nC-1, nC).long().cuda()))
		cls_ = cls_.view(nB*nA, nC, nH*nW).transpose(1, 2).contiguous().view(nB*nA*nH*nW, nC)
		'''
		Part2: Get predict results 
		'''
		pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
		grid_x = torch.linspace(0, nW-1, nW).repeat(nH, 1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
		grid_y = torch.linspace(0, nH-1, nH).repeat(nW, 1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
		# anchor_step = 2
		anchor_step = len(self.options.get('anchors')) // self.options.get('num_anchors')
		assert anchor_step == 2
		anchor_w = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([0])).cuda()
		anchor_h = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([1])).cuda()
		anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
		anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
		pred_boxes[0] = x.data + grid_x
		pred_boxes[1] = y.data + grid_y
		pred_boxes[2] = torch.exp(w.data) * anchor_w
		pred_boxes[3] = torch.exp(h.data) * anchor_h
		pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))
		'''
		Part3: get targets, prepare for calculating loss.
		'''
		noobject_scale = self.options.get('noobject_scale')
		object_scale = self.options.get('object_scale')
		sil_thresh = self.options.get('sil_thresh')
		seen = self.options.get('seen')
		max_object = self.options.get('max_object')
		nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes=pred_boxes, 
																									target=target.data, 
																									anchors=anchors, 
																									num_anchors=nA, 
																									num_classes=nC, 
																									nH=nH, 
																									nW=nW, 
																									noobject_scale=noobject_scale, 
																									object_scale=object_scale, 
																									sil_thresh=sil_thresh, 
																									seen=seen,
																									max_object=max_object)
		cls_mask = (cls_mask == 1)
		# conf > 0.25 as the proposals.
		nProposals = int((conf > 0.25).sum().data[0])
		# the targets.
		tx = Variable(tx.cuda())
		ty = Variable(ty.cuda())
		tw = Variable(tw.cuda())
		th = Variable(th.cuda())
		tconf = Variable(tconf.cuda())
		tcls = Variable(tcls.view(-1)[cls_mask].long().cuda())
		# the masks
		coord_mask = Variable(coord_mask.cuda())
		conf_mask = Variable(conf_mask.cuda().sqrt())
		cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC).cuda())
		cls_ = cls_[cls_mask].view(-1, nC)
		'''
		Part4: calculate loss.
		'''
		coord_scale = self.options.get('coord_scale')
		class_scale = self.options.get('class_scale')
		loss_x = coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
		loss_y = coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
		loss_w = coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
		loss_h = coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
		loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
		loss_cls = class_scale * nn.CrossEntropyLoss(size_average=False)(cls_, tcls)
		loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
		return loss