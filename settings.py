import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Defining Parameters
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(128*128*3)]  # 128x128 RGB image flattened

# Image formats allowed
allowed_image_type = ['.png', '.jpg', '.jpeg']

@st.cache_data
def initialize_data():
    if 'operators_db' not in st.session_state:
        st.session_state.operators_db = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
    return st.session_state.operators_db

def add_data_db(df_operators_details):
    try:
        st.session_state.operators_db = pd.concat([st.session_state.operators_db, df_operators_details], ignore_index=True)
        st.session_state.operators_db.drop_duplicates(subset=COLS_INFO, keep='first', inplace=True)
        st.session_state.operators_db.reset_index(inplace=True, drop=True)
        return True
    except Exception as e:
        st.error(f"Error adding data to database: {e}")
        return False

def attendance(id, name):
    if 'attendance_db' not in st.session_state:
        st.session_state.attendance_db = pd.DataFrame(columns=["id", "operator_name", "Timing"])
    
    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendance_temp = pd.DataFrame({"id": [id], "operator_name": [name], "Timing": [dtString]})
    
    st.session_state.attendance_db = pd.concat([st.session_state.attendance_db, df_attendance_temp], ignore_index=True)

@st.cache_data
def load_attendance_data():
    if 'attendance_db' not in st.session_state:
        st.session_state.attendance_db = pd.DataFrame(columns=["id", "operator_name", "Timing"])
    return st.session_state.attendance_db

def view_attendace():
    df_attendance = load_attendance_data()
    df_attendance = df_attendance.sort_values(by='Timing', ascending=False)
    df_attendance.reset_index(drop=True, inplace=True)
    st.write(df_attendance)

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))
