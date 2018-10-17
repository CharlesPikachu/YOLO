'''
Author: 
	Charles
Function:
	set options.
'''


# for yolo1
yolo1_options = {
					'info': 'yolo1_options',
					'max_object': 50,
					'backupdir': './backup',
					'trainSet': '/home/zcjin/voc_train.txt',
					'testSet': '/home/zcjin/2007_test.txt',
					'trainlabpth': None,
					'testlabpth': None,
					'clsnamesfile': './names/coco.names',
					'gpus': '0, 1',
					'ngpus': 2,
					'use_cuda': True,
					'num_workers': 4,
					'is_multiscale': False,
					'by_stride': False,
					'weightfile': './weights/yolov1.weights',
					'cfgfile': './cfg/yolov1.cfg',
					'save_interval': 10,
					'conf_thresh': 0.25,
					'nms_thresh': 0.4,
					'iou_thresh': 0.5,
					'jitter': 0.2,
					'mode': 'test'
				}


# for yolo2
yolo2_options = {
					'info': 'yolo2_options',
					'max_object': 50,
					'backupdir': './backup',
					'trainSet': '/home/zcjin/voc_train.txt',
					'testSet': '/home/zcjin/2007_test.txt',
					'trainlabpth': None,
					'testlabpth': None,
					'clsnamesfile': './names/coco.names',
					'gpus': '0, 1',
					'ngpus': 2,
					'use_cuda': True,
					'num_workers': 4,
					'is_multiscale': True,
					'by_stride': False,
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
yolo3_options = {
					'info': 'yolo3_options',
					'max_object': 50,
					'backupdir': './backup',
					'trainSet': '/home/zcjin/voc_train.txt',
					'testSet': '/home/zcjin/2007_test.txt',
					'trainlabpth': None,
					'testlabpth': None,
					'clsnamesfile': './names/coco.names',
					'gpus': '1',
					'ngpus': 2,
					'use_cuda': True,
					'num_workers': 4,
					'is_multiscale': True,
					'by_stride': True,
					'weightfile': './weights/yolov3.weights',
					'cfgfile': './cfg/yolov3.cfg',
					'save_interval': 10,
					'conf_thresh': 0.25,
					'nms_thresh': 0.4,
					'iou_thresh': 0.5,
					'jitter': 0.3,
					'mode': 'test'
				}