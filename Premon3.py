import uuid
import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from streamlit_option_menu import option_menu
from settings import *

# Use st.cache_data for data loading functions
@st.cache_data
def load_database_data():
    return initialize_data()

# Use st.cache_resource for resource-intensive operations
@st.cache_resource
def process_image(image_array):
    face_locations = face_recognition.face_locations(image_array)
    encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)
    return face_locations, encodesCurFrame

def main():
    st.set_page_config(page_title="Presence Monitoring Webapp", page_icon="👥", layout="wide")
    
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
        for path in [OPERATORS_DB, OPERATORS_HISTORY]:
            if os.path.exists(path):
                for file in os.listdir(path):
                    os.remove(os.path.join(path, file))
            else:
                os.makedirs(path)
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
        
        # Save operator history
        with open(os.path.join(OPERATORS_HISTORY, f'{operator_id}.jpg'), 'wb') as file:
            file.write(img_file_buffer.getbuffer())
        st.success('Image Saved Successfully!')

        face_locations, encodesCurFrame = process_image(image_array)

        if face_locations:
            process_faces(image_array, face_locations, encodesCurFrame, operator_id)
        else:
            st.error('No human face detected.')

def process_faces(image_array, face_locations, encodesCurFrame, operator_id):
    database_data = load_database_data()
    face_encodings = database_data[COLS_ENCODE].values
    dataframe = database_data[COLS_INFO]

    for face_idx, (face_encode, (top, right, bottom, left)) in enumerate(zip(encodesCurFrame, face_locations)):
        dataframe['distance'] = face_recognition.face_distance(face_encodings, face_encode)
        dataframe['similarity'] = dataframe['distance'].apply(face_distance_to_conf)
        dataframe_new = dataframe[dataframe['similarity'] > 0.5].sort_values(by="similarity", ascending=False).head(1)

        if not dataframe_new.empty:
            name_operator = dataframe_new.iloc[0]['Name']
            attendance(operator_id, name_operator)
            draw_face_box(image_array, left, top, right, bottom, name_operator)
            st.success(f"{name_operator} has been recognized and logged.")
        else:
            st.error(f'No Match Found for the given Similarity Threshold for face#{face_idx}')
            attendance(operator_id, 'Unknown')
            draw_face_box(image_array, left, top, right, bottom, "Unknown")

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
    
    with open(os.path.join(OPERATORS_DB, f'{face_name}.jpg'), 'wb') as file:
        file.write(img_file_buffer.getbuffer())

    face_locations, encodesCurFrame = process_image(image_array)

    if encodesCurFrame:
        df_new = pd.DataFrame(data=encodesCurFrame, columns=COLS_ENCODE)
        df_new[COLS_INFO] = face_name
        df_new = df_new[COLS_INFO + COLS_ENCODE].copy()
        add_data_db(df_new)
        st.success(f"Face data for {face_name} added to the database.")
    else:
        st.error("No face detected in the image. Please try again with a clear face image.")

if __name__ == "__main__":
    main()
