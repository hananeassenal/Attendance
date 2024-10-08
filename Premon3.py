import streamlit as st
import cv2
import numpy as np
import pandas as pd
import uuid
from streamlit_option_menu import option_menu
from settings import *
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Presence Monitoring Webapp", page_icon="👥", layout="wide")

# Initialize the database at the start of the application
initialize_data()

def detect_face(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a binary mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assumed to be the face)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [(x, y, w, h)]
    return []

def main():
    st.markdown(
        f"""
        <div style="background-color:#bddc6d;padding:12px">
        <h1 style="color:white;text-align:center;">Presence Monitoring Webapp</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("About")
    st.sidebar.info("This webapp monitors the presence of operators in a smart factory using OpenCV and Streamlit")

    if st.sidebar.button('Clear all data'):
        if 'operators_db' in st.session_state:
            st.session_state.operators_db = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        if 'attendance_db' in st.session_state:
            st.session_state.attendance_db = pd.DataFrame(columns=["id", "operator_name", "Timing"])
        st.sidebar.success("All data cleared!")

    selected_menu = option_menu(None, 
                                ['Operator Validation', 'View Operator History', 'Add to Database'], 
                                icons=['camera', "clock-history", 'person-plus'], 
                                menu_icon="cast", 
                                default_index=0, 
                                orientation="horizontal")

    if selected_menu == 'Operator Validation':
        operator_validation()
    elif selected_menu == 'View Operator History':
        view_attendace()
    elif selected_menu == 'Add to Database':
        add_to_database()

def operator_validation():
    operator_id = uuid.uuid1()
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        faces = detect_face(image_array)

        if faces:
            process_faces(image_array, faces, operator_id)
        else:
            st.error('No human face detected.')

def process_faces(image_array, faces, operator_id):
    database_data = st.session_state.operators_db

    for (x, y, w, h) in faces:
        face_img = image_array[y:y+h, x:x+w]
        face_encoding = cv2.resize(face_img, (128, 128)).flatten()

        # Compare with database
        similarities = database_data[COLS_ENCODE].apply(lambda row: np.dot(face_encoding, row) / (np.linalg.norm(face_encoding) * np.linalg.norm(row)), axis=1)
        best_match = similarities.idxmax()
        
        if similarities[best_match] > 0.7:  # Adjust threshold as needed
            name_operator = database_data.loc[best_match, 'Name']
            attendance(operator_id, name_operator)
            draw_face_box(image_array, x, y, x+w, y+h, name_operator)
            st.success(f"{name_operator} has been recognized and logged.")
        else:
            st.error(f'No Match Found for the given Similarity Threshold')
            attendance(operator_id, 'Unknown')
            draw_face_box(image_array, x, y, x+w, y+h, "Unknown")

    st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), width=720)

def draw_face_box(image, left, top, right, bottom, name):
    cv2.rectangle(image, (left, top), (right, bottom), COLOR_DARK, 2)
    cv2.rectangle(image, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
    cv2.putText(image, f"#{name}", (left + 5, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, .55, COLOR_WHITE, 1)

def add_to_database():
    col1, col2, col3 = st.columns(3)
    face_name = col1.text_input('Name:', '')
    pic_option = col2.radio('Upload Picture', options=["Upload a Picture", "Click a picture"])

    if pic_option == 'Upload a Picture':
        img_file_buffer = col3.file_uploader('Upload a Picture', type=allowed_image_type)
    else:
        img_file_buffer = col3.camera_input("Click a picture")

    if img_file_buffer and face_name and st.button('Click to Save!'):
        process_new_face(img_file_buffer, face_name)

def process_new_face(img_file_buffer, face_name):
    file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)
    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    faces = detect_face(image_array)

    if faces:
        (x, y, w, h) = faces[0]
        face_img = image_array[y:y+h, x:x+w]
        face_encoding = cv2.resize(face_img, (128, 128)).flatten()

        # Ensure the face encoding has the correct number of elements
        if len(face_encoding) != len(COLS_ENCODE):
            st.error(f"Face encoding length ({len(face_encoding)}) does not match expected length ({len(COLS_ENCODE)})")
            return

        # Create a dictionary with the correct column names and values
        face_data = {col: val for col, val in zip(COLS_ENCODE, face_encoding)}
        face_data[COLS_INFO[0]] = face_name  # Add the name column

        df_new = pd.DataFrame([face_data])

        if add_data_db(df_new):
            st.success(f"Face data for {face_name} added to the database.")
        else:
            st.error("Failed to add face data to the database.")
    else:
        st.error("No face detected in the image. Please try again with a clear face image.")

if __name__ == "__main__":
    main()
