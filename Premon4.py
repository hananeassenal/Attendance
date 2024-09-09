import cv2
import streamlit as st
import numpy as np
import uuid
import os
import pandas as pd
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from settings import *

st.set_page_config(page_title="Presence Monitoring Webapp", page_icon="👥", layout="wide")

# Static Data
user_color = '#bddc6d'
title_webapp = "Presence Monitoring Webapp"

@st.cache_data
def load_database():
    return initialize_data()

@st.cache_resource
def process_image(image_array):
    face_locations = face_recognition.face_locations(image_array)
    encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)
    return face_locations, encodesCurFrame

def main():
    st.markdown(
        f"""
        <div style="background-color:{user_color};padding:12px">
        <h1 style="color:white;text-align:center;">{title_webapp}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("About")
    st.sidebar.info("This webapp monitors the presence of operators in a smart factory using 'Face Recognition' and Streamlit")

    if st.sidebar.button('Clear all data'):
        clear_data()

    selected_menu = option_menu(None,
                                ['Operator Validation', 'View Operator History', 'Add to Database'],
                                icons=['camera', "clock-history", 'person-plus'],
                                menu_icon="cast", default_index=0, orientation="horizontal")

    if selected_menu == 'Operator Validation':
        operator_validation()
    elif selected_menu == 'View Operator History':
        view_attendace()
    elif selected_menu == 'Add to Database':
        add_to_database()

def clear_data():
    for path in [OPERATORS_DB, OPERATORS_HISTORY]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    st.sidebar.success("All data cleared!")

def operator_validation():
    operator_id = uuid.uuid1()
    last_detection_time = {}

    cap = cv2.VideoCapture(0)
    
    stframe = st.empty()
    stop_button = st.button('Stop')

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations, encodesCurFrame = process_image(image_array)

        for idx, (top, right, bottom, left) in enumerate(face_locations):
            cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
            cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
            cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, .55, COLOR_WHITE, 1)

        stframe.image(image_array, channels="RGB")

        if face_locations:
            process_detected_faces(face_locations, encodesCurFrame, operator_id, last_detection_time)

        if st.button('Stop'):
            break

    cap.release()

def process_detected_faces(face_locations, encodesCurFrame, operator_id, last_detection_time):
    database_data = load_database()
    face_encodings = database_data[COLS_ENCODE].values
    dataframe = database_data[COLS_INFO]

    for face_encode in encodesCurFrame:
        dataframe['distance'] = face_recognition.face_distance(face_encodings, face_encode)
        dataframe['similarity'] = dataframe['distance'].apply(face_distance_to_conf)
        
        match = dataframe[dataframe['similarity'] > 0.5].sort_values(by="similarity", ascending=False).head(1)

        if not match.empty:
            name_operator = match.iloc[0]['Name']
            current_time = datetime.now()

            if name_operator not in last_detection_time or \
                    (current_time - last_detection_time[name_operator]) > timedelta(minutes=1):
                attendance(operator_id, name_operator)
                last_detection_time[name_operator] = current_time
                st.success(f"{name_operator} has been recognized and logged.")
            else:
                st.info(f"{name_operator} was already logged less than a minute ago.")
        else:
            attendance(operator_id, 'Unknown')
            st.warning("Unknown person detected, please update the database.")

def add_to_database():
    col1, col2, col3 = st.columns(3)
    face_name = col1.text_input('Name:', '')
    pic_option = col2.radio('Upload Picture', options=["Upload a Picture", "Click a picture"])

    if pic_option == 'Upload a Picture':
        img_file_buffer = col3.file_uploader('Upload a Picture', type=ALLOWED_IMAGE_TYPE)
    else:
        img_file_buffer = col3.camera_input("Click a picture")

    if img_file_buffer and face_name and st.button('Save to Database'):
        process_new_face(img_file_buffer, face_name)

def process_new_face(img_file_buffer, face_name):
    file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)
    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with open(os.path.join(OPERATORS_DB, f'{face_name}.jpg'), 'wb') as file:
        file.write(img_file_buffer.getbuffer())

    face_locations, encodesCurFrame = process_image(image_array)

    if encodesCurFrame:
        df_new = pd.DataFrame(data=encodesCurFrame, columns=COLS_ENCODE)
        df_new[COLS_INFO] = face_name
        df_new = df_new[COLS_INFO + COLS_ENCODE].copy()
        if add_data_db(df_new):
            st.success(f"{face_name} added to the database successfully!")
        else:
            st.error("Failed to add to database. Please try again.")
    else:
        st.error("No face detected in the image. Please try again with a clear face image.")

if __name__ == "__main__":
    main()
