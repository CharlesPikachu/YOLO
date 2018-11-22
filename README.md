# YOLO
```sh
A pytorch implementation of YOLOv1-v3.  
Project only supports python3.x.
```

# To do
- [ ] Test the model of YOLOv1.
- [ ] Train the model of YOLOv1.
- [x] Test the model of YOLOv2.
- [x] Train the model of YOLOv2.
- [x] Test the model of YOLOv3.
- [x] Train the model of YOLOv3.
- [x] Data augmentation.
- [x] Using k-means to generate the priors.
- [x] Evaluate the model including mAP, precision, recall, etc.
- [ ] Data of VOC format -> YOLO format.

# Dependency
- torch 0.3.1
- opencv-python
- torchvision
- numpy
- pillow
- argparse

# Train
## Prepare
#### VOC -> YOLO
```sh
Data of VOC format(lxml) -> YOLO format(txt).
You can use the script in ./TOOL/voc2yolo.py to complete the conversion work.
```
- Usage
	- modify Line<xx~xx> according to your needs
	- run "python3 voc2yolo.py"
#### Get good priors
```sh
Run k-means clustering on the dimensions of bounding boxes to get good priors for our model.
You can use the script in ./TOOL/genPriors/genPriors.py to get the good priors.
```
- Usage
	- modify the options.json according to your needs
	- run "python3 genPriors.py"
## YOLOV1
```sh
preparing...
```
## YOLOV2
#### Step1
```sh
pip install -r requirements.txt
```
#### Step2
```sh
modify the config.py-yolo2_options
```
- set mode -> train
- set weightfile -> [darknet19_448.conv.23](https://pjreddie.com/media/files/darknet19_448.conv.23)
- set clsnamesfile -> coco.names, voc.names, etc.
- set trainSet, testSet, cfgfile, gpus, ngpus, etc.
#### Step3
```sh
run "python3 train.py --version yolo2"
```
## YOLOV3
#### Step1
```sh
pip install -r requirements.txt
```
#### Step2
```sh
modify the config.py-yolo3_options
```
- set mode -> train
- set weightfile -> [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
- set clsnamesfile -> coco.names, voc.names, etc.
- set trainSet, testSet, cfgfile, gpus, ngpus, etc.
#### Step3
```sh
run "python3 train.py --version yolo3"
```

# Test
## YOLOV1
```sh
preparing
```
## YOLOV2
#### Step1
```sh
modify the config.py-yolo2_options
```
- set mode - test
- set weightfile -> [yolov2.weights](https://pjreddie.com/media/files/yolov2.weights)
- set clsnamesfile -> coco.names, voc.names, etc.
#### Step2
```sh
run "python3 detector.py --version yolo2"
```
## YOLOV3
#### Step1
```sh
modify the config.py-yolo3_options
```
- set mode - test
- set weightfile -> [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- set clsnamesfile -> coco.names, voc.names, etc.
#### Step2
```sh
run "python3 detector.py --version yolo3"
```

# Eval
```sh
preparing
```

# Reference
- [marvis](https://github.com/marvis/pytorch-yolo2)
- [yolov1](https://arxiv.org/abs/1506.02640)
- [yolov2](https://arxiv.org/abs/1612.08242)
- [yolov3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)