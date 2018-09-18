'''
Author: 
	Charles
Function:
	set options.
'''


# for yolo1
yolo1_options = {}


# for yolo2
yolo2_options = {
					'info': 'yolo2_options',
					'max_object': 50,
					'stride': 32,
					'backupdir': './backup',
					'trainSet': '/home/zcjin/voc_train.txt',
					'testSet': '/home/zcjin/2007_test.txt',
					'labpth': None,
					'clsnamesfile': './names/coco.names',
					'gpus': '0, 1',
					'ngpus': 2,
					'use_cuda': True,
					'num_workers': 4,
					'is_multiscale': True,
					'weightfile': './weights/yolov2.weights',
					'cfgfile': './cfg/yolov2.cfg',
					'save_interval': 10,
					'conf_thresh': 0.25,
					'nms_thresh': 0.4,
					'iou_thresh': 0.5,
					'jitter': 0.3,
					'mode': 'test'
				}


# for yolo3
yolo3_options = {}