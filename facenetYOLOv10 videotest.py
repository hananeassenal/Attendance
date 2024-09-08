# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:01:40 2024

@author: Khadija
"""

import os
import pickle
import cv2
from ultralytics import YOLO
from keras_facenet import FaceNet
from PIL import Image as Img
from numpy import asarray, expand_dims
import numpy as np

# Load the face recognition database
database_file = "C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\data.pkl"
with open(database_file, "rb") as myfile:
    database = pickle.load(myfile)

# Load YOLO model for face detection
model_path = "C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\runs\\detect\\train2\\weights\\last.pt"
model = YOLO(model_path)  # Load the custom YOLO model

# Load FaceNet model
facenet_model = FaceNet()

# Load video file
video_path = "C:\\Users\\Khadija\\Downloads\\The Office Skit from the Emmys.mp4"
cap = cv2.VideoCapture(video_path)  # Replace with the path to your video file

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object to save output video
output_path = "C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\FNYLoutput_video.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

threshold = 0.5  # Confidence threshold for detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Crop and preprocess the detected face
            face = frame[int(y1):int(y2), int(x1):int(x2)]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Img.fromarray(face)
            face = face.resize((160, 160))
            face = asarray(face)

            face = expand_dims(face, axis=0)
            signature = facenet_model.embeddings(face)

            # Compare the detected face with the database
            min_dist = float("inf")
            identity = "Unknown"

            for name, db_embeddings in database.items():
                for db_embedding in db_embeddings:
                    dist = np.linalg.norm(db_embedding - signature)
                    if dist < min_dist:
                        min_dist = dist
                        identity = name

            # Display the result
            label = f"{identity}" if min_dist < 0.5 else "Unknown"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Save the frame with the detections
    out.write(frame)

    # Display the resulting frame (optional)
    cv2.imshow('Video', frame)

    # Check for 'q' key press to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_path}")
