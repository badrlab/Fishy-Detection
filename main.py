from subprocess import check_output

from utils import run_yolo

use_colab = False

# Clone darknet repo
check_output("git clone https://github.com/AlexeyAB/darknet", shell=True)

# Build Darknet
check_output("cd darknet", shell=True)
# change makefile to have GPU and OPENCV enabled
check_output("sed -i 's/OPENCV=0/OPENCV=1/' Makefile", shell=True)
check_output("sed -i 's/GPU=0/GPU=1/' Makefile", shell=True)
check_output("sed -i 's/CUDNN=0/CUDNN=1/' Makefile", shell=True)
check_output("sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile", shell=True)
# check vuda version
check_output("/usr/local/cuda/bin/nvcc --version", shell=True)
# build darknet
check_output("make", shell=True)

# Download pretrained fish yolov3 weights
#check_output("wget https://www.dropbox.com/s/gmw2774nrsw7ovk/yolov3-obj_30000.weights?dl=0", shell=True)
#check_output("mv 'yolov3-obj_30000.weights?dl=0' yolov3fish.weights", shell=True)

# run darknet detection on test images
threshold = 0.5
dataset_config_file = "configs/data.data"
model_config_file = "configs/yolov3fish.cfg"
model_weights = "weights/yolov3fish.weights"
img_path = "data/tuna1.jpg"

run_yolo(img_path=img_path,
         threshold=threshold,
         dataset_config_file=dataset_config_file,
         model_config_file=model_config_file,
         model_weights=model_weights)

# Visualization

# Training

# Testing
