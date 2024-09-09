import os
import pathlib
import datetime
import pandas as pd
import cv2
import numpy as np
import streamlit as st

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Streamlit static path
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

# Create necessary directories
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
OUT_DIR = (STREAMLIT_STATIC_PATH / "output")
OPERATORS_DB = os.path.join(ROOT_DIR, "operators_database")
OPERATORS_HISTORY = os.path.join(ROOT_DIR, "operators_history")

for dir_path in [DOWNLOADS_PATH, LOG_DIR, OUT_DIR, OPERATORS_DB, OPERATORS_HISTORY]:
    os.makedirs(dir_path, exist_ok=True)

# Defining Parameters
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(128)]  # 128-dimensional face encoding

# Database
DATA_PATH = OPERATORS_DB
FILE_DB = 'operators_db.csv'
FILE_HISTORY = 'operators_history.csv'

# Image formats allowed
allowed_image_type = ['.png', '.jpg', '.jpeg']

@st.cache_data
def initialize_data():
    db_path = os.path.join(DATA_PATH, FILE_DB)
    if os.path.exists(db_path):
        return pd.read_csv(db_path)
    else:
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(db_path, index=False)
        return df

def add_data_db(df_operators_details):
    db_path = os.path.join(DATA_PATH, FILE_DB)
    try:
        df_all = pd.read_csv(db_path)
        df_all = pd.concat([df_all, df_operators_details], ignore_index=True)
        df_all.drop_duplicates(subset=COLS_INFO, keep='first', inplace=True)
        df_all.reset_index(inplace=True, drop=True)
        df_all.to_csv(db_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error adding data to database: {e}")
        return False

def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))

def attendance(id, name):
    f_p = os.path.join(OPERATORS_HISTORY, FILE_HISTORY)
    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendance_temp = pd.DataFrame({"id": [id], "operator_name": [name], "Timing": [dtString]})
    
    if not os.path.isfile(f_p):
        df_attendance_temp.to_csv(f_p, index=False)
    else:
        df_attendance = pd.read_csv(f_p)
        df_attendance = pd.concat([df_attendance, df_attendance_temp], ignore_index=True)
        df_attendance.to_csv(f_p, index=False)

@st.cache_data
def load_attendance_data():
    f_p = os.path.join(OPERATORS_HISTORY, FILE_HISTORY)
    if not os.path.isfile(f_p):
        return pd.DataFrame(columns=["id", "operator_name", "Timing"])
    return pd.read_csv(f_p)

def view_attendace():
    df_attendance = load_attendance_data()
    df_attendance = df_attendance.sort_values(by='Timing', ascending=False)
    df_attendance.reset_index(drop=True, inplace=True)
    st.write(df_attendance)

    if df_attendance.shape[0] > 0:
        selected_img = st.selectbox('Search Image using ID', options=['None'] + list(df_attendance['id']))
        if selected_img != 'None':
            avail_files = [file for file in os.listdir(OPERATORS_HISTORY)
                           if file.endswith(tuple(allowed_image_type)) and file.startswith(str(selected_img))]
            if avail_files:
                st.image(os.path.join(OPERATORS_HISTORY, avail_files[0]))
            else:
                st.write("No image found for the selected ID.")
