# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:04:41 2024

@author: Khadija
"""

import os
import pickle
import cv2
from keras_facenet import FaceNet
from PIL import Image as Img
from numpy import asarray, expand_dims
import numpy as np

# Load the face recognition database
database_file = "C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\data.pkl"
with open(database_file, "rb") as myfile:
    database = pickle.load(myfile)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\haarcascade_frontalface_default.xml")

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (required for Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x1, y1, width, height) in faces:
        x2, y2 = x1 + width, y1 + height

        # Crop and preprocess the detected face
        face = frame[y1:y2, x1:x2]
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
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
