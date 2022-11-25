#!/bin/sh
cd / && git clone https://github.com/JunnYu/mish-cuda
cd /yolo && git clone https://github.com/WongKinYiu/ScaledYOLOv4.git
cd /mish-cuda && python setup.py build install
cp /yolo/yolov4-p6-tensorrt/gen_wts.py /yolo/ScaledYOLOv4/gen_wts.py
cd /yolo/ScaledYOLOv4 && python gen_wts.py weights/yolov4-p6.weights
cp /yolo/ScaledYOLOv4/yolov4-p6.wts /yolo/yolov4-p6-tensorrt
cd /yolo/yolov4-p6-tensorrt && mkdir build
cd /yolo/yolov4-p6-tensorrt/build && cmake ..
cd /yolo/yolov4-p6-tensorrt/build && make
/yolo/yolov4-p6-tensorrt/build/yolov4-p6 -s 1.0 1.0
/yolo/yolov4-p6-tensorrt/build/yolov4-p6 -d /yolo/yolov4-p6-tensorrt/samples