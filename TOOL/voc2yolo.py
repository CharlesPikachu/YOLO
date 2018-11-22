'''
Author:
	Charles
Function:
	VOC format -> YOLO format.
'''
import os
import xml.etree.ElementTree as ET


# some options which you should modify according to your needs.
options = {
			'clsnamesfile': '/home/zcjin/YOLO/names/voc.names',
			'ann_dir': '/home/zcjin/VOCdevkit/VOC2007/Annotations',
			'save_dir': None
		}


'''
Function:
	Parse xml files.
'''
def parse_xml(xmlpath, clsnames):
	xmlfile = open(xmlpath)
	tree = ET.parse(xmlfile)
	root = tree.getroot()
	size = root.find('size')
	width = int(size.find('width').text)
	height = int(size.find('height').text)
	filename = root.find('filename').text
	all_instances = [filename.split('.')[0] + '.txt']
	for obj in root.iter('object'):
		difficult = obj.find('difficult').text
		clsname = obj.find('name').text
		if clsname not in clsnames or int(difficult) == 1:
			continue
		cls_id = clsnames.index(clsname)
		bndbox = obj.find('bndbox')
		box = [float(bndbox.find('xmin').text), float(bndbox.find('xmax').text), float(bndbox.find('ymin').text), float(bndbox.find('ymax').text)]
		x = ((box[0] + box[1]) / 2.0) / width
		y = ((box[2] + box[3]) / 2.0) / height
		w = (box[1] - box[0]) / width
		h = (box[3] - box[2]) / height
		new_box = [x, y, w, h]
		outstr = str(cls_id) + ' ' + ' '.join([str(i) for i in new_box]) + '\n'
		all_instances.append(outstr)
	return all_instances


'''
Function:
	Call this function to complete voc -> yolo.
'''
def main():
	with open(options.get('clsnamesfile'), 'r') as f:
		clsnames = f.readlines()
	clsnames = [clsname.strip('\n') for clsname in clsnames]
	ann_dir = options.get('ann_dir')
	save_dir = options.get('save_dir')
	if not save_dir:
		tmp = ann_dir.split('/')[:-1]
		save_dir = '/'.join(tmp)
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	for xmlpath in os.listdir(ann_dir):
		res = parse_xml(os.path.join(ann_dir, xmlpath), clsnames)
		with open(os.path.join(save_dir, res[0]), 'w') as f:
			for i in res[1:]:
				f.write(i)


if __name__ == '__main__':
	main()