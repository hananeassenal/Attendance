import os, pathlib
import streamlit as st
import os, datetime, json, sys, pathlib, shutil
import pandas as pd
import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
########################################################################################################################
# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

## We create a downloads directory within the streamlit static asset directory and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
if not LOG_DIR.is_dir():
    LOG_DIR.mkdir()

OUT_DIR = (STREAMLIT_STATIC_PATH / "output")
if not OUT_DIR.is_dir():
    OUT_DIR.mkdir()

OPERATORS_DB = os.path.join(ROOT_DIR, "operators_database")
# st.write(OPERATOR_DB)

if not os.path.exists(OPERATORS_DB):
    os.mkdir(OPERATORS_DB)

OPERATORS_HISTORY = os.path.join(ROOT_DIR, "operators_history")
# st.write(OPERATOR_HISTORY)

if not os.path.exists(OPERATORS_HISTORY):
    os.mkdir(OPERATORS_HISTORY)
########################################################################################################################
## Defining Parameters

COLOR_DARK  = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO   = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(128)]

## Database
data_path       = OPERATORS_DB
file_db         = 'operators_db.csv'         ## To store user information
file_history    = 'operators_history.csv'    ## To store operators history information

## Image formats allowed
allowed_image_type = ['.png', 'jpg', '.jpeg']
################################################### Defining Function ##############################################
def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        # st.info('Database Found!')
        df = pd.read_csv(os.path.join(data_path, file_db))

    else:
        # st.info('Database Not Found!')
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(os.path.join(data_path, file_db), index=False)

    return df

#################################################################
def add_data_db(df_operators_details):
    try:
        df_all = pd.read_csv(os.path.join(data_path, file_db))

        if not df_all.empty:
            df_all = pd.concat([df_all, df_operators_details], ignore_index=True)
            df_all.drop_duplicates(keep='first', inplace=True)
            df_all.reset_index(inplace=True, drop=True)
            df_all.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Details Added Successfully!')
        else:
            df_operators_details.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Initiated Data Successfully!')

    except Exception as e:
        st.error(e)

#################################################################
# convert opencv BRG to regular RGB mode
def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

#################################################################
def findEncodings(images):
    encode_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list

#################################################################
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power(
            (linear_val - 0.5) * 2, 0.2))

#################################################################
def attendance(id, name):
    f_p = os.path.join(OPERATORS_HISTORY, file_history)
    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendace_temp = pd.DataFrame(data={
        "id": [id],
        "operator_name": [name],
        "Timing": [dtString]
    })

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
    else:
        df_attendace = pd.read_csv(f_p)
        # Use pd.concat instead of append
        df_attendace = pd.concat([df_attendace, df_attendace_temp], ignore_index=True)
        df_attendace.to_csv(f_p, index=False)


#################################################################
def view_attendace():
    # Define the path to the history file
    f_p = os.path.join(OPERATORS_HISTORY, file_history)

    # Initialize an empty DataFrame if the file does not exist
    if not os.path.isfile(f_p):
        df_attendace_temp = pd.DataFrame(columns=["id", "operator_name", "Timing"])
        df_attendace_temp.to_csv(f_p, index=False)
        st.write("No attendance records found.")
        return
    else:
        # Load the existing DataFrame from the CSV file
        df_attendace_temp = pd.read_csv(f_p)

    # Sort the DataFrame by 'Timing' in descending order
    df_attendace = df_attendace_temp.sort_values(by='Timing', ascending=False)
    df_attendace.reset_index(drop=True, inplace=True)

    # Display the DataFrame
    st.write(df_attendace)

    # Check if there are any records to process
    if df_attendace.shape[0] > 0:
        # Select an ID from the DataFrame
        selected_img = st.selectbox('Search Image using ID', options=['None'] + list(df_attendace['id']))

        # Find files that match the selected ID
        avail_files = [file for file in list(os.listdir(OPERATORS_HISTORY))
                       if file.endswith(tuple(allowed_image_type)) and file.startswith(selected_img)]

        # Display the selected image if available
        if len(avail_files) > 0:
            selected_img_path = os.path.join(OPERATORS_HISTORY, avail_files[0])
            st.image(Image.open(selected_img_path))
        else:
            st.write("No image found for the selected ID.")


########################################################################################################################