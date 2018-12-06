'''
Author:
	Charles
Function:
	eval the model of yoloV1-yoloV3.
'''
import torch
import config
from utils.utils import *
from torchvision import transforms
from torch.autograd import Variable
from dataset.dataset import myDataset


'''
Function:
	class for eval including precision, recall, etc.
'''
class evalModel():
	def __init__(self, **kwargs):
		self.options = kwargs
		self.use_cuda = kwargs.get('use_cuda')
		kgs = {'num_workers': kwargs.get('num_workers'), 'pin_memory': True} if self.use_cuda else {}
		self.test_loader = torch.utils.data.DataLoader(
								myDataset(root=[kwargs.get('testSet'), kwargs.get('testlabpth')],
										  shape=(kwargs.get('init_width'), kwargs.get('init_height')),
										  shuffle=False,
										  transform=transforms.Compose([transforms.ToTensor(),]),
										  is_train=False,
										  num_workers=kwargs.get('num_workers'),
										  is_multiscale=kwargs.get('is_multiscale'),
										  max_object=kwargs.get('max_object')),
								shuffle=False,
								batch_size=kwargs.get('batch_size'),
								**kgs)
	# eval the model
	def eval(self, model):
		model.eval()
		if self.options.get('ngpus') > 1:
			cur_model = model.module
		else:
			cur_model = model
		cur_model.options['mode'] = 'test'
		num_classes = int(cur_model.blocks[-1]['classes'])
		anchors = [float(i) for i in cur_model.blocks[-1]['anchors'].split(',')]
		num_anchors = int(cur_model.blocks[-1]['num'])
		conf_thresh = self.options.get('conf_thresh')
		by_stride = self.options.get('by_stride')
		iou_thresh = self.options.get('iou_thresh')
		anchor_masks = []
		for block in cur_model.blocks:
			if block['layer_type'] == 'yolo':
				anchor_mask = [int(i) for i in block['mask'].split(',')]
				anchor_masks.append(anchor_mask)
		# count
		total = 0.0
		proposals = 0.0
		correct = 0.0
		for batch_idx, (data, target) in enumerate(self.test_loader):
			if self.options.get('use_cuda'):
				data = data.cuda()
			data = Variable(data, volatile=True)
			output = model(data)
			yolo_type = self.options.get('yolo_type')
			if yolo_type == 'yolo1':
				boxes = get_boxes_yolo1(output,
										conf_thresh=conf_thresh,
										num_classes=num_classes,
										num_anchors=num_anchors,
										width=cur_model.blocks[0]['width'],
										height=cur_model.blocks[0]['height'],
										stride=cur_model.det_strides[0])[0]
			elif yolo_type == 'yolo2':
				boxes = get_boxes_yolo2(output,
										conf_thresh=conf_thresh,
										num_classes=num_classes,
										anchors=anchors,
										num_anchors=num_anchors,
										stride=cur_model.det_strides[0],
										by_stride=by_stride)[0]
			elif yolo_type == 'yolo3':
				boxes = get_boxes_yolo3(output,
										conf_thresh=conf_thresh,
										num_classes=num_classes,
										anchors=anchors,
										num_anchors=num_anchors,
										anchor_masks=anchor_masks,
										stride=cur_model.det_strides,
										by_stride=by_stride)
				bboxes = []
				for bs in boxes:
					bboxes += bs
				boxes = bboxes
			else:
				raise ValueError('Unknow yolo_type <%s>...' % yolo_type)
			for b in range(target.size(0)):
				boxes = nms(boxes, self.options.get('nms_thresh'))
				truths = target[b].view(-1, 5)
				num_gts = self.__truths_len(truths)
				total += num_gts
				for i in range(len(boxes)):
					if boxes[i][4] > conf_thresh:
						proposals += 1
				for k in range(num_gts):
					# [x, y, w, h, det_conf, cls_conf, cls_id]
					box_gt = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], 1.0, 1.0, truths[k][0]]
					best_iou = 0
					best_kk = -1
					for kk in range(len(boxes)):
						iou = bbox_iou(box_gt, boxes[kk], x1y1x2y2=False)
						if iou > best_iou:
							best_kk = kk
							best_iou = iou
					if best_iou > iou_thresh and boxes[best_kk][6] == box_gt[6]:
						correct += 1
		eps = 1e-6
		precision = 1.0 * correct / (proposals + eps)
		recall = 1.0 * correct / (total + eps)
		fscore = 2.0 * precision * recall / (precision + recall + eps)
		logging("[precision]: %f, [recall]: %f, [fscore]: %f" % (precision, recall, fscore))
		cur_model.options['mode'] = 'train'
	def __truths_len(self, truths):
		i = -1
		while True:
			i += 1
			if truths[i][1] == 0:
				return i