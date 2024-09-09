import streamlit as st
import uuid
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import pandas as pd
import face_recognition
from settings import *

st.set_page_config(page_title="Presence Monitoring Webapp", page_icon="👥", layout="wide")

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
        <div style="background-color:#bddc6d;padding:12px">
        <h1 style="color:white;text-align:center;">Presence Monitoring Webapp</h1>
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
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        image_array_copy = image_array.copy()

        save_operator_history(operator_id, img_file_buffer)

        face_locations, encodesCurFrame = process_image(image_array)

        if face_locations:
            process_faces(image_array, image_array_copy, face_locations, encodesCurFrame, operator_id)
        else:
            st.error('No human face detected.')

def save_operator_history(operator_id, img_file_buffer):
    with open(os.path.join(OPERATORS_HISTORY, f'{operator_id}.jpg'), 'wb') as file:
        file.write(img_file_buffer.getbuffer())
    st.success('Image Saved Successfully!')

def process_faces(image_array, image_array_copy, face_locations, encodesCurFrame, operator_id):
    for idx, (top, right, bottom, left) in enumerate(face_locations):
        cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
        cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
        cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, .55, COLOR_WHITE, 1)

    st.image(BGR_to_RGB(image_array), width=720)

    col1, col2 = st.columns(2)
    face_idxs = col1.multiselect("Select face#", range(len(face_locations)), default=range(len(face_locations)))
    similarity_threshold = col2.slider('Select Threshold for Similarity', min_value=0.0, max_value=1.0, value=0.5)

    if st.button('Process Selected Faces') and face_idxs:
        process_selected_faces(face_idxs, face_locations, encodesCurFrame, image_array_copy, similarity_threshold, operator_id)

def process_selected_faces(face_idxs, face_locations, encodesCurFrame, image_array_copy, similarity_threshold, operator_id):
    database_data = load_database()
    face_encodings = database_data[COLS_ENCODE].values
    dataframe = database_data[COLS_INFO]

    for face_idx in face_idxs:
        face_encode = encodesCurFrame[face_idx]
        dataframe['distance'] = face_recognition.face_distance(face_encodings, face_encode)
        dataframe['similarity'] = dataframe['distance'].apply(face_distance_to_conf)
        
        dataframe_new = dataframe[dataframe['similarity'] > similarity_threshold].sort_values(by="similarity", ascending=False).head(1)

        if not dataframe_new.empty:
            name_operator = dataframe_new.iloc[0]['Name']
            attendance(operator_id, name_operator)
            draw_face_box(image_array_copy, face_locations[face_idx], name_operator)
        else:
            st.error(f'No Match Found for the given Similarity Threshold for face#{face_idx}')
            attendance(operator_id, 'Unknown')

    st.image(BGR_to_RGB(image_array_copy), width=720)

def draw_face_box(image, face_location, name):
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), COLOR_DARK, 2)
    cv2.rectangle(image, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
    cv2.putText(image, f"#{name}", (left + 5, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, .55, COLOR_WHITE, 1)

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
