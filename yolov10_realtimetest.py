# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:19:06 2024

@author: Khadija
"""

import os
from ultralytics import YOLO
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)  # '0' is the ID of the default webcam, change if you have multiple cameras

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame to get the video properties
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    cap.release()
    exit()

H, W, _ = frame.shape

# Define the codec and create VideoWriter object to save output video
video_path_out = "C:\\Users\\Khadija\\Videos\\webcam_out.mp4"
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (W, H))

model_path = "C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\runs\\detect\\train2\\weights\\last.pt"

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Write the frame into the output video file
    out.write(frame)

    # Check for 'q' key press to break the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Detected 'q' key press. Exiting loop.")
        break

# When everything is done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
print("Resources released and windows closed.")
