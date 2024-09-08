import cv2
import streamlit as st
import numpy as np
import uuid
import shutil
import os
import pandas as pd
import time
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from settings import *

#######################################################
# Disable Warnings
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_option('deprecation.showfileUploaderEncoding', False)
#######################################################

# Static Data
user_color = '#bddc6d'
title_webapp = "Presence Monitoring Webapp"

html_temp = f"""
<div style="background-color:{user_color};padding:12px">
<h1 style="color:white;text-align:center;">{title_webapp}</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Defining Static Paths
if st.sidebar.button('Click to Clear out all the data'):
    shutil.rmtree(OPERATORS_DB, ignore_errors=True)
    os.mkdir(OPERATORS_DB)
    shutil.rmtree(OPERATORS_HISTORY, ignore_errors=True)
    os.mkdir(OPERATORS_HISTORY)

if not os.path.exists(OPERATORS_DB):
    os.mkdir(OPERATORS_DB)

if not os.path.exists(OPERATORS_HISTORY):
    os.mkdir(OPERATORS_HISTORY)

# Dictionary to store the last detection time of each operator
last_detection_time = {}

########################################################################################
def main():
    st.sidebar.header("About")
    st.sidebar.info("This webapp monitors the presence of operators in a smart factory using 'Face Recognition' and Streamlit")

    selected_menu = option_menu(None,
                                ['Operator Validation', 'View Operator History', 'Add to Database'],
                                icons=['camera', "clock-history", 'person-plus'],
                                menu_icon="cast", default_index=0, orientation="horizontal")

    if selected_menu == 'Operator Validation':
        operator_id = uuid.uuid1()

        # Capture video from webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            # Convert the image from BGR to RGB
            image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_array_copy = image_array.copy()

            # Detect faces in the loaded image
            face_locations = face_recognition.face_locations(image_array)
            encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)

            # Generating Rectangle Red box over the Image
            for idx, (top, right, bottom, left) in enumerate(face_locations):
                cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

            # Display the image with bounding boxes
            st.image(image_array, channels="RGB")

            max_faces = len(face_locations)

            if max_faces > 0:
                dataframe_new = pd.DataFrame()
                rois = []

                for face_idx in range(max_faces):
                    roi = image_array_copy[face_locations[face_idx][0]:face_locations[face_idx][2],
                                           face_locations[face_idx][3]:face_locations[face_idx][1]].copy()
                    rois.append(roi)

                    # initial database for known faces
                    database_data = initialize_data()
                    face_encodings = database_data[COLS_ENCODE].values
                    dataframe = database_data[COLS_INFO]

                    faces = face_recognition.face_encodings(roi)

                    if len(faces) < 1:
                        st.error(f'Please Try Again for face#{face_idx}!')
                    else:
                        face_to_compare = faces[0]
                        dataframe['distance'] = face_recognition.face_distance(face_encodings, face_to_compare)
                        dataframe['distance'] = dataframe['distance'].astype(float)
                        dataframe['similarity'] = dataframe.distance.apply(lambda distance: f"{face_distance_to_conf(distance):0.2}")
                        dataframe['similarity'] = dataframe['similarity'].astype(float)

                        dataframe_new = dataframe.drop_duplicates(keep='first')
                        dataframe_new.reset_index(drop=True, inplace=True)
                        dataframe_new.sort_values(by="similarity", ascending=True)

                        dataframe_new = dataframe_new[dataframe_new['similarity'] > 0.5].head(1)
                        dataframe_new.reset_index(drop=True, inplace=True)

                        if dataframe_new.shape[0] > 0:
                            name_operator = dataframe_new.loc[0, 'Name']
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

                # Break the loop if you want to stop capturing
                if st.button('Stop'):
                    break

        cap.release()

    elif selected_menu == 'View Operator History':
        view_attendace()

    if selected_menu == 'Add to Database':
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
#######################################################
