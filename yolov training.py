# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:58:26 2024

@author: Khadija
"""

from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10n.yaml")

# Train the model
model.train(data="C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\config.yaml", epochs=100, imgsz=640, batch=10)