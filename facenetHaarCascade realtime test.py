import os
from keras_facenet import FaceNet
from PIL import Image as Img
from numpy import asarray, expand_dims
import cv2
import pickle

# Folder containing images for the database
folder = "C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\Photos for face recognition"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\haarcascade_frontalface_default.xml")

# Load FaceNet model
facenet_model = FaceNet()

# Initialize an empty database dictionary
database = {}

# Loop through all images in the folder
for filename in os.listdir(folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        name = os.path.splitext(filename)[0]  # Extract the name (without extension)
        path = os.path.join(folder, filename)
        
        # Load the image using OpenCV
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to load image {path}.")
            continue
        
        # Detect faces using Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            x1, y1, width, height = faces[0]  # Assuming the first detected face is the one we want
        else:
            print(f"No face detected in {filename}.")
            continue

        # Crop and preprocess the face
        x2, y2 = x1 + width, y1 + height
        face = image[y1:y2, x1:x2]

        # Convert the face to RGB (if not already) and resize to 160x160
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Img.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)

        # Get face embedding
        face = expand_dims(face, axis=0)
        signature = facenet_model.embeddings(face)

        # Add the embedding to the database
        if name in database:
            database[name].append(signature)
        else:
            database[name] = [signature]

# Save the database to a .pkl file
database_file = "C:\\Users\\Khadija\\OneDrive\\Desktop\\smart factory\\data.pkl"
with open(database_file, "wb") as myfile:
    pickle.dump(database, myfile)

print(f"Database created and saved to {database_file}.")
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

# Initialize webcam
cap = cv2.VideoCapture(0)  # '0' is the ID of the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Check for 'q' key press to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
