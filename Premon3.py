#######################################################
import uuid ## random id generator
from streamlit_option_menu import option_menu
from settings import *
#######################################################
## Disable Warnings
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('deprecation.showfileUploaderEncoding', False)
################################################### Defining Static Data ###############################################


user_color      = '#bddc6d'
title_webapp    = "Presence Monitoring Webapp"

html_temp = f"""
            <div style="background-color:{user_color};padding:12px">
            <h1 style="color:white;text-align:center;">{title_webapp}
            </h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)

###################### Defining Static Paths ###################4
if st.sidebar.button('Click to Clear out all the data'):
    ## Clearing operator Database
    shutil.rmtree(OPERATORS_DB, ignore_errors=True)
    os.mkdir(OPERATORS_DB)
    ## Clearing operator History
    shutil.rmtree(OPERATORS_HISTORY, ignore_errors=True)
    os.mkdir(OPERATORS_HISTORY)

if not os.path.exists(OPERATORS_DB):
    os.mkdir(OPERATORS_DB)

if not os.path.exists(OPERATORS_HISTORY):
    os.mkdir(OPERATORS_HISTORY)
# st.write(OPERATORS_HISTORY)
########################################################################################
def main():
    ###################################################
    st.sidebar.header("About")
    st.sidebar.info("This webapp monitors the presence of operators in a smart factory using 'Face Recognition' and Streamlit")
    ###################################################

    selected_menu = option_menu(None, 
                                ['Operator Validation', 'View Operator History', 'Add to Database'], 
                                icons=['camera', "clock-history", 'person-plus'], 
                                menu_icon="cast", 
                                default_index=0, 
                                orientation="horizontal")

    if selected_menu == 'Operator Validation':
        operator_id = uuid.uuid1()
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image_array_copy = image_array.copy()

            # Save operator history
            with open(os.path.join(OPERATORS_HISTORY, f'{operator_id}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                st.success('Image Saved Successfully!')

            face_locations = face_recognition.face_locations(image_array)
            encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)

            if len(face_locations) > 0:
                rois = []
                flag_show = False
                dataframe_new = pd.DataFrame()

                for idx, (top, right, bottom, left) in enumerate(face_locations):
                    rois.append(image_array[top:bottom, left:right].copy())
                    cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                    cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                st.image(BGR_to_RGB(image_array), width=720)

                for face_idx, face_encode in enumerate(encodesCurFrame):
                    database_data = initialize_data()
                    face_encodings = database_data[COLS_ENCODE].values
                    dataframe = database_data[COLS_INFO]

                    dataframe['distance'] = face_recognition.face_distance(face_encodings, face_encode)
                    dataframe['similarity'] = dataframe['distance'].apply(lambda dist: face_distance_to_conf(dist))
                    dataframe_new = dataframe[dataframe['similarity'] > 0.5].sort_values(by="similarity", ascending=False).head(1)

                    if not dataframe_new.empty:
                        name_operator = dataframe_new.iloc[0]['Name']
                        attendance(operator_id, name_operator)

                        cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
                        cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                        cv2.putText(image_array_copy, f"#{name_operator}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)
                        flag_show = True
                    else:
                        st.error(f'No Match Found for the given Similarity Threshold for face#{face_idx}')
                        attendance(operator_id, 'Unknown')

                if flag_show:
                    st.image(BGR_to_RGB(image_array_copy), width=720)
            else:
                st.error('No human face detected.')

    if selected_menu == 'View Operator History':
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
