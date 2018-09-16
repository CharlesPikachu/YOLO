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
					'max_object': 50,
					'stride': 32,
					'backupdir': './backup',
					'trainSet': '',
					'testSet': '',
					'namespath': '',
					'gpu': '0, 1',
					'ngpu': 2,
					'use_cuda': True,
					'num_workers': 4,
					'weightfile': '',
					'cfgfile': './cfg/yolov2.cfg',
					'save_interval': 10,
					'conf_thresh': 0.25,
					'nms_thresh': 0.4,
					'iou_thresh': 0.5
				}


# for yolo3
yolo3_options = {}