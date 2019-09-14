import numpy as np
import argparse
import time
import cv2
import os

# Argument parser for input and parameters.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	            help="Path to an input image")
ap.add_argument("-y", "--yolo", required=True,
	            help="Base path to YOLO folder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	            help="Minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	            help="Threshold when applying non-maxima suppression")

# Turning args into keyword arguments using vars() function:
args = vars(ap.parse_args())

# Loading COCO class labels the YOLO model was trained on:
labels_path = os.path.sep.join([arg["yolo"], "coco.names"])
labels = open(labels_path).read().strip().split("\n")

# Initializing a list of colors to represent each possible class label.
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(labels), 3),
                           dtype="uint8")

# Defining paths to YOLO weights and model config:
weights_path = os.path.sep.join([args["yolo"], "yolov3.weights"])
config_path = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load YOLO object detector trained on COCO dataset w/ 80 classes.
print("[INFO] loading YOLO from disk")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)  # OpenCV >= 3.4.2

# Load input image and get its spatial dimenstions.
image = cv2.imread(args["image"])
h, w = image.shape[:2]

# Define layer names - only the *output* layer.
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Construct a blob from the input image and perform a forward pass
# of the YOLO object detector, outputs bounding boxed and associated
# probabilities.
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	                         swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layer_outputs = net.forward(ln)
end = time.time()

# Show timing info on YOLO:
print("[INFO] YOLO took {:6f} seconds".format(end - start))

# Init lists of detected bounding boxes, confidences and class IDs.
boxes = []
confidences = []
class_ids = []
