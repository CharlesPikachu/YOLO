'''
Author:
	Charles
Function:
	Run k-means clustering on the dimensions of bounding boxes to get good priors for our model. 
'''
import os
import json
import random
import numpy as np


'''
Function:
	Save priors.
'''
def save_priors(priors):
	with open('resultPriors.txt', 'w') as f:
		for prior in priors:
			f.write('%s %s\n' % (str(prior[0]), str(prior[1])))
	print('[INFO]:The priors saved to resultPriors.txt successfully...')


'''
Function:
	Print priors.
'''
def print_priors(centeranchors, width, height, is_save=True):
	outstring = '[Priors]: \n'
	priors = centeranchors.copy()
	widths = priors[:, 0]
	sorted_inds = np.argsort(widths)
	if is_save:
		priorsList = []
	for i in sorted_inds:
		tmp_w, tmp_h = int(round(priors[i, 0] * width)), int(round(priors[i, 1] * height))
		if is_save:
			priorsList.append([tmp_w, tmp_h])
		outstring += str(tmp_w) + ', ' + str(tmp_h) + '\n'
	if is_save:
		save_priors(priorsList)
	print(outstring[0: -1])


'''
Function:
	Compute IoU.
'''
def IoU(wh, centeranchors):
	w, h = wh
	results = []
	for ca in centeranchors:
		c_w, c_h = ca
		if c_w >= w and c_h >= h:
			result = w * h / (c_w * c_h)
		elif c_w >= w and c_h <= h:
			result = w * c_h / (w * h + c_w * c_h - w * c_h)
		elif c_w <= w and c_h >= h:
			result = c_w * h / (w * h + c_w * c_h - c_w * h)
		else:
			result = c_w * c_h / (w * h)
		results.append(result)
	return np.array(results)


'''
Function:
	Compute average IoU.
'''
def avgIoU(whs, centeranchors):
	ann_num, anchor_dim = whs.shape
	sum_ = 0
	for i in range(ann_num):
		sum_ += max(IoU(whs[i], centeranchors))
	return sum_ / ann_num


'''
Function:
	Kmeans.
'''
def kmeans(whs, num_priors):
	ann_num, anchor_dim = whs.shape
	iteration = 0
	prev_assignments = np.ones(ann_num) * (-1)
	prev_distances = np.zeros((ann_num, num_priors))
	inds = [random.randrange(ann_num) for i in range(num_priors)]
	centeranchors = whs[inds]
	all_distances = []
	while True:
		distances = []
		iteration += 1
		for i in range(ann_num):
			d = 1 - IoU(whs[i], centeranchors)
			distances.append(d)
		distances = np.array(distances)
		print('[Iteration {}]: distances = {}'.format(iteration, np.sum(np.abs(prev_distances - distances))))
		all_distances.append(np.sum(np.abs(prev_distances - distances)))
		assignments = np.argmin(distances, axis=1)
		if (assignments == prev_assignments).all():
			return centeranchors
		centeranchors_sum = np.zeros((num_priors, anchor_dim), np.float)
		for i in range(ann_num):
			centeranchors_sum[assignments[i]] += whs[i]
		for j in range(num_priors):
			centeranchors[j] = centeranchors_sum[j] / (np.sum(assignments == j) + 1e-6)
		prev_assignments = assignments.copy()
		prev_distances = distances.copy()


'''
Function:
	Call this function to get the good priors for our model training.
'''
def run():
	with open('./options.json', 'r') as f:
		options = json.load(f)
	num_priors = int(options['num_priors'])
	width = int(options['width'])
	height = int(options['height'])
	txtlable_dir = options['txtlable_dir']
	allWH_relative = []
	all_labels = sorted(os.listdir(txtlable_dir))
	for label in all_labels:
		fl = open(os.path.join(txtlable_dir, label), 'r')
		gts = fl.readlines()
		for gt in gts:
			gt = gt.strip()
			if not gt:
				break
			gt = gt.split(' ')
			assert len(gt) == 5
			relative_w = float(gt[3])
			relative_h = float(gt[4])
			allWH_relative.append([relative_w, relative_h])
	allWH_relative = np.array(allWH_relative)
	centeranchors = kmeans(allWH_relative, num_priors)
	print('[Num_Anchors]: %s, [Average IoU]: %.2f' % (num_priors, avgIoU(allWH_relative, centeranchors)))
	print_priors(centeranchors, width, height, is_save=True)


if __name__ == '__main__':
	run()