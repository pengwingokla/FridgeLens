# import libraries
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import gc

# import the newly trained model
classifier = YOLO("D:\HackAI\FridgeLens\computer_vision/best.pt")

# test phase
results = classifier.predict(source="D:\HackAI\FridgeLens\computer_vision\IMG_6792.jpg")
results[0].show()