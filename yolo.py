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
