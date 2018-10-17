'''
Author:
	Charles
Function:
	YoloV1-V3 based detector.
'''
import cv2
import sys
import torch
import config
from utils.utils import *
from nets.darknet import Darknet


'''
Function:
	A demo for yolo detector.
Input:
	-yolo_type: the type of yolo.
'''
def Demo(yolo_type='yolo2'):
	# for yoloV1
	if yolo_type == 'yolo1':
		options = config.yolo1_options
		weightfile = options.get('weightfile')
		clsnamesfile = options.get('clsnamesfile')
		use_cuda = options.get('use_cuda')
		conf_thresh = options.get('conf_thresh')
		nms_thresh = options.get('nms_thresh')
		model = Darknet(options)
		width = int(model.blocks[0]['width'])
		height = int(model.blocks[0]['height'])
		num_classes = int(model.blocks[-1]['classes'])
		num_anchors = int(model.blocks[-1]['num'])
		if (not weightfile) or (not clsnamesfile):
			print('[Error]: You should assign the weightfile and clsnamesfile in config.py-yolo1_options...')
			sys.exit(-1)
		model.load_weights(options.get('weightfile'))
		print('[INFO]: Loading weights from %s... Done!' % weightfile)
		class_names = load_class_names(clsnamesfile)
		if use_cuda:
			model.cuda()
		capture = cv2.VideoCapture(0)
		if not capture.isOpened():
			print('[Error]:Unable to open camera...')
			sys.exit(-1)
		while True:
			res, img = capture.read()
			if res:
				sized = cv2.resize(img, (width, height))
				model.eval()
				if isinstance(sized, Image.Image):
					sized = image2torch(sized)
				elif type(sized) == np.ndarray:
					sized = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
				else:
					print("[Error]: unknow image type...")
					sys.exit(-1)
				if use_cuda:
					sized = sized.cuda()
				sized = torch.autograd.Variable(sized)
				output = model(sized)
				output = output.data
				boxes = get_boxes_yolo1(output,
										conf_thresh=conf_thresh,
										num_classes=num_classes,
										num_anchors=num_anchors,
										width=width,
										height=height,
										stride=model.det_strides[0])[0]
				bboxes = nms(boxes, nms_thresh)
				draw_img = plot_boxes_cv2(img, bboxes, class_names)
				cv2.imshow('yolo1', draw_img)
				cv2.waitKey(1)
			else:
				sys.exit(-1)
	# for yoloV2
	elif yolo_type == 'yolo2':
		options = config.yolo2_options
		weightfile = options.get('weightfile')
		clsnamesfile = options.get('clsnamesfile')
		use_cuda = options.get('use_cuda')
		conf_thresh = options.get('conf_thresh')
		nms_thresh = options.get('nms_thresh')
		by_stride = options.get('by_stride')
		model = Darknet(options)
		width = int(model.blocks[0]['width'])
		height = int(model.blocks[0]['height'])
		anchors = [float(i) for i in model.blocks[-1]['anchors'].split(',')]
		num_classes = int(model.blocks[-1]['classes'])
		num_anchors = int(model.blocks[-1]['num'])
		if (not weightfile) or (not clsnamesfile):
			print('[Error]: You should assign the weightfile and clsnamesfile in config.py-yolo2_options...')
			sys.exit(-1)
		model.load_weights(options.get('weightfile'))
		print('[INFO]: Loading weights from %s... Done!' % weightfile)
		class_names = load_class_names(clsnamesfile)
		if use_cuda:
			model.cuda()
		capture = cv2.VideoCapture(0)
		if not capture.isOpened():
			print('[Error]:Unable to open camera...')
			sys.exit(-1)
		while True:
			res, img = capture.read()
			if res:
				sized = cv2.resize(img, (width, height))
				model.eval()
				if isinstance(sized, Image.Image):
					sized = image2torch(sized)
				elif type(sized) == np.ndarray:
					sized = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
				else:
					print("[Error]: unknow image type...")
					sys.exit(-1)
				if use_cuda:
					sized = sized.cuda()
				sized = torch.autograd.Variable(sized)
				output = model(sized)
				output = output.data
				boxes = get_boxes_yolo2(output,
										conf_thresh=conf_thresh,
										num_classes=num_classes,
										anchors=anchors,
										num_anchors=num_anchors,
										stride=model.det_strides[0],
										by_stride=by_stride)[0]
				bboxes = nms(boxes, nms_thresh)
				draw_img = plot_boxes_cv2(img, bboxes, class_names)
				cv2.imshow('yolo2', draw_img)
				cv2.waitKey(1)
			else:
				sys.exit(-1)
	# for yoloV3
	elif yolo_type == 'yolo3':
		options = config.yolo3_options
		weightfile = options.get('weightfile')
		clsnamesfile = options.get('clsnamesfile')
		use_cuda = options.get('use_cuda')
		conf_thresh = options.get('conf_thresh')
		nms_thresh = options.get('nms_thresh')
		by_stride = options.get('by_stride')
		model = Darknet(options)
		width = int(model.blocks[0]['width'])
		height = int(model.blocks[0]['height'])
		anchors = [float(i) for i in model.blocks[-1]['anchors'].split(',')]
		anchor_masks = []
		for block in model.blocks:
			if block['layer_type'] == 'yolo':
				anchor_mask = [int(i) for i in block['mask'].split(',')]
				anchor_masks.append(anchor_mask)
		num_classes = int(model.blocks[-1]['classes'])
		num_anchors = int(model.blocks[-1]['num'])
		if (not weightfile) or (not clsnamesfile):
			print('[Error]: You should assign the weightfile and clsnamesfile in config.py-yolo3_options...')
			sys.exit(-1)
		model.load_weights(options.get('weightfile'))
		print('[INFO]: Loading weights from %s... Done!' % weightfile)
		class_names = load_class_names(clsnamesfile)
		if use_cuda:
			model.cuda()
		capture = cv2.VideoCapture(0)
		if not capture.isOpened():
			print('[Error]:Unable to open camera...')
			sys.exit(-1)
		while True:
			res, img = capture.read()
			if res:
				sized = cv2.resize(img, (width, height))
				model.eval()
				if isinstance(sized, Image.Image):
					sized = image2torch(sized)
				elif type(sized) == np.ndarray:
					sized = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
				else:
					print("[Error]: unknow image type...")
					sys.exit(-1)
				if use_cuda:
					sized = sized.cuda()
				sized = torch.autograd.Variable(sized)
				output = model(sized)
				output = output
				boxes = get_boxes_yolo3(output,
										conf_thresh=conf_thresh,
										num_classes=num_classes,
										anchors=anchors,
										num_anchors=num_anchors,
										anchor_masks=anchor_masks,
										stride=model.det_strides,
										by_stride=by_stride)
				bboxes = []
				for bs in boxes:
					bboxes += bs
				bboxes = nms(bboxes, nms_thresh)
				draw_img = plot_boxes_cv2(img, bboxes, class_names)
				cv2.imshow('yolo3', draw_img)
				cv2.waitKey(1)
			else:
				sys.exit(-1)
	else:
		raise ValueError('yolo_type should be in [yolo1, yolo2, yolo3]...')




if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--version', help='Choose the version of yolo.')
	args = parser.parse_args()
	Demo(yolo_type=args.version)