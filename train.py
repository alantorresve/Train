import os
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n-pose.yaml')

    # Train the model
    results = model.train(data='data.yaml', epochs=500, imgsz=640, device='cuda', single_cls=True, batch=8)


