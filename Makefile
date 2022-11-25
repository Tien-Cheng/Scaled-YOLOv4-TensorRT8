start_build:
	nvidia-docker run --name build_yolov4 -it -v $(CURDIR):/yolo --shm-size=64g nvcr.io/nvidia/pytorch:22.08-py3
