# README

This is a fork of [tjuskyzhang/Scaled-YOLOv4-TensorRT](https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT) and [WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4.git), which implements Scaled YOLOv4 model in TensorRT. Made this fork as was having trouble getting the model to build properly on the latest version of TensorRT.


## Modifications
- Shoddily hacked together code to get it to build properly in TensortRT version 8. Seems to work, but I did get some warnings about subnormal values at half (FP16) precision.
- Fixed the `gen_wts.py` script which is supposed to accept an argument for the location of the `.pt` weights, but is actually hard coded.

## How to build
1. Install dependencies with poetry
2. Run `dvc pull` to pull the model weights (or download them from the ScaledYOLOv4 repo)
3. Run `make start_build` to launch a Nvidia docker container linked to this project
4. Within the interactive shell, cd to `/yolo` and run `build.sh`
5. Model weights will be in `yolov4-p6-tensorrt/build`
   - yolov4-p6.engine
   - libmyplugins.so is needed to add support certain layers, and must be loaded in Triton Inference


## References
- https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large
- https://github.com/wang-xinyu/tensorrtx
- https://github.com/thomasbrandon/mish-cuda
- https://github.com/wang-xinyu/tensorrtx/pull/1064/