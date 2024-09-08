import uuid
import streamlit as st
from streamlit_option_menu import option_menu
import os
import shutil
import numpy as np
import cv2
import face_recognition
import pandas as pd
from settings import *
from datetime import datetime

#######################################################
# Utility Functions

def attendance(operator_id, name_operator):
    # Update operator history with timestamp, operator ID, and operator name
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("operator_history.csv", "a") as f:
        f.write(f"{timestamp},{operator_id},{name_operator}\n")

def initialize_data():
    # Load your database with operator face encodings and info
    return pd.read_csv('face_database.csv')

def BGR_to_RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def face_distance_to_conf(distance):
    return 1 - distance  # Adjust based on your similarity logic

#######################################################
# Static Settings

user_color = '#bddc6d'
title_webapp = "Presence Monitoring Webapp"

html_temp = f"""
            <div style="background-color:{user_color};padding:12px">
            <h1 style="color:white;text-align:center;">{title_webapp}
            </h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)

# Clear data buttons
if st.sidebar.button('Click to Clear out all the data'):
    shutil.rmtree(OPERATORS_DB, ignore_errors=True)
    os.mkdir(OPERATORS_DB)
    shutil.rmtree(OPERATORS_HISTORY, ignore_errors=True)
    os.mkdir(OPERATORS_HISTORY)

if not os.path.exists(OPERATORS_DB):
    os.mkdir(OPERATORS_DB)

if not os.path.exists(OPERATORS_HISTORY):
    os.mkdir(OPERATORS_HISTORY)

#######################################################
# Main Function
def main():
    st.sidebar.header("About")
    st.sidebar.info("This webapp monitors the presence of operators in a smart factory using 'Face Recognition' and Streamlit")

    selected_menu = option_menu(None, 
                                ['Operator Validation', 'View Operator History', 'Add to Database'], 
                                icons=['camera', "clock-history", 'person-plus'], 
                                menu_icon="cast", 
                                default_index=0, 
                                orientation="horizontal")

    if selected_menu == 'Operator Validation':
        st.subheader("Operator Validation")

        # Initialize webcam for real-time capture
        cap = cv2.VideoCapture(0)
        operator_id = uuid.uuid1()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error('Failed to capture frame from webcam.')
                break

            image_array = frame.copy()

            # Detect faces in the frame
            face_locations = face_recognition.face_locations(image_array)
            encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)

            if len(face_locations) > 0:
                flag_show = False
                for idx, (top, right, bottom, left) in enumerate(face_locations):
                    cv2.rectangle(image_array, (left, top), (right, bottom), (0, 0, 255), 2)

                st.image(BGR_to_RGB(image_array), width=720)

                # Compare each face encoding with database
                for face_encode in encodesCurFrame:
                    database_data = initialize_data()
                    face_encodings = database_data['encoding'].values
                    dataframe = database_data[['Name', 'other_columns']]

                    dataframe['distance'] = face_recognition.face_distance(face_encodings, face_encode)
                    dataframe['similarity'] = dataframe['distance'].apply(lambda dist: face_distance_to_conf(dist))
                    dataframe_new = dataframe[dataframe['similarity'] > 0.5].sort_values(by="similarity", ascending=False).head(1)

                    if not dataframe_new.empty:
                        name_operator = dataframe_new.iloc[0]['Name']
                        attendance(operator_id, name_operator)
                        flag_show = True
                    else:
                        attendance(operator_id, 'Unknown')

                if flag_show:
                    st.image(BGR_to_RGB(image_array), width=720)
            else:
                st.error('No human face detected.')

        cap.release()

    elif selected_menu == 'View Operator History':
        st.subheader("Operator History")
        view_attendace()

    elif selected_menu == 'Add to Database':
        st.subheader("Add New Operator to Database")

        col1, col2, col3 = st.columns(3)
        face_name = col1.text_input('Name:', '')
        pic_option = col2.radio('Upload Picture', options=["Upload a Picture", "Click a picture"])

        if pic_option == 'Upload a Picture':
            img_file_buffer = col3.file_uploader('Upload a Picture', type=allowed_image_type)
        elif pic_option == 'Click a picture':
            img_file_buffer = col3.camera_input("Click a picture")

        if img_file_buffer and face_name and st.button('Click to Save!'):
            file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            with open(os.path.join(OPERATORS_DB, f'{face_name}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())

            face_locations = face_recognition.face_locations(image_array)
            encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)

            df_new = pd.DataFrame(data=encodesCurFrame, columns=COLS_ENCODE)
            df_new[COLS_INFO] = face_name
            df_new = df_new[COLS_INFO + COLS_ENCODE].copy()
            add_data_db(df_new)

#######################################################
if __name__ == "__main__":
    main()
