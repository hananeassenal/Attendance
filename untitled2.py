# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:56:45 2024

@author: Khadija
"""

import os
import cv2
import face_recognition
import numpy as np
import datetime
from ultralytics import YOLO

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Path to your known faces directory
known_faces_dir = "path_to_known_faces_directory"

for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(filename)[0])

# Initialize webcam
cap = cv2.VideoCapture(0)  # '0' is the default webcam

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
video_path_out = "webcam_out.mp4"
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (W, H))

model_path = "C:\\Users\\Khadija\\OneDrive\\Desktop\\runs\\detect\\train2\\weights\\last.pt"

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5
attendance_log = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    face_locations = []
    face_images = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            face_locations.append((int(y1), int(x2), int(y2), int(x1)))
            face_images.append(frame[int(y1):int(y2), int(x1):int(x2)])

    for face_location, face_image in zip(face_locations, face_images):
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face_image)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                attendance_log[name] = attendance_log.get(name, 0) + 1
                cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (0, 255, 0), 2)
                cv2.putText(frame, name, (face_location[3], face_location[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

# Save attendance log to file
date_str = datetime.datetime.now().strftime("%Y-%m-%d")
with open(f"attendance_{date_str}.txt", "w") as file:
    for name, count in attendance_log.items():
        file.write(f"{name}: {count}\n")
