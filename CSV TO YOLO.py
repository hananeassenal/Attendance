# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:01:24 2024

@author: Khadija
"""

#Let's convert the csv annotations to yolo format
import pandas as pd
import os

#Reading the csv files
os.chdir("C:\\Program Files\\OIDv4_ToolKit\\OID\\csv_folder")
print("Working directory is",os.getcwd())
classes_data=pd.read_csv("C:\\Program Files\\OIDv4_ToolKit\\OID\\csv_folder\\class-descriptions-boxable.csv",header=None)
classes_data.head()

classes=["Human face"]


#get the strings corresponding to the class name and store it inside the list class_strings
class_string=[]
for i in classes:
    req_classes=classes_data.loc[classes_data[1]==i]
    string=req_classes.iloc[0][0]
    class_string.append(string)
print(class_string)

# Define the path to the CSV file
csv_path = "C:\\Program Files\\OIDv4_ToolKit\\OID\\csv_folder\\train-annotations-bbox.csv"

# Initialize an empty DataFrame to hold the filtered data
annotation_data = pd.DataFrame()

# Define columns to read
usecols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]

# Read the CSV file in chunks
chunksize = 10000  # Number of rows per chunk
for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
    annotation_data = pd.concat([annotation_data, chunk], ignore_index=True)

# Display the first few rows of the DataFrame
print(annotation_data.head())

#get only records with matching classes
filtered_class_data= annotation_data.loc[annotation_data['LabelName'].isin(class_string)].copy()
filtered_class_data.head()

#Add new columns required for yolo format
filtered_class_data["classNumber"]=""
filtered_class_data["center x"]=""
filtered_class_data["center y"]=""
filtered_class_data["width"]=""
filtered_class_data["height"]=""

#iterate all the class strings and assign a class number according to the order they are appearing in the list
for i in range(len(class_string)):
    #store the result to classNumber
    filtered_class_data.loc[filtered_class_data["LabelName"]==class_string[i],"classNumber"]=i
#calculate center x and center y values
filtered_class_data['center x']=(filtered_class_data["XMax"]+filtered_class_data["XMin"])/2
filtered_class_data['center y']=(filtered_class_data["YMax"]+filtered_class_data["YMin"])/2

#calculate width and height values
filtered_class_data['width']=filtered_class_data["XMax"]-filtered_class_data["XMin"]
filtered_class_data['height']=filtered_class_data["YMax"]-filtered_class_data["YMin"]

#Generate the dataframe with YOLO required values
YOLO_values=filtered_class_data.loc[:,["ImageID", "classNumber", "center x", "center y", "width", "height"]].copy()
YOLO_values.head()


# Path to the directory where images are stored
images_path = "C:\\Users\Khadija\\Documents\\YOLOAnnotations"

# Directory to save the text files
output_path = "C:\\Users\\Khadija\\Documents\\YOLOAnnotations"
os.makedirs(output_path, exist_ok=True)

# Change the current directory to the images directory
os.chdir(images_path)

# Iterate through all files in the directory
for current_dir, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".jpg"):
            # Extract only the title of the image file to use for the corresponding txt file
            image_title = f[:-4]
            
            # Select rows where ImageID matches the image title
            YOLO_file = YOLO_values.loc[YOLO_values["ImageID"] == image_title]
            
            # Create a copy of the DataFrame with required columns
            df = YOLO_file.loc[:, ['classNumber', 'center x', 'center y', 'width', 'height']].copy()
            
            # Define the save path for the text file
            save_path = os.path.join(output_path, image_title + '.txt')
            
            # Generate a text file containing the required data in YOLO format
            df.to_csv(save_path, header=False, index=False, sep=' ')

# Display a message indicating completion
print("YOLO format txt files created successfully.")


# Paths
images_path = "C:\\Users\\Khadija\\Documents\\YOLOAnnotations"
output_path = "C:\\Users\\Khadija\\Documents\\YOLOAnnotations"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Create list of image paths
path_list = []
for current_dir, dirs, files in os.walk(images_path):
    for f in files:
        if f.endswith(".jpg"):
            file_loc = os.path.join(images_path, f)
            path_list.append(file_loc + '\n')

# Split data into training (80%) and testing (20%)
path_list_test = path_list[:int(len(path_list) * 0.20)]
path_list_train = path_list[int(len(path_list) * 0.20):]

# Create train.txt file
train_file_path = os.path.join(output_path, 'train.txt')
with open(train_file_path, 'w') as train_file:
    for path in path_list_train:
        train_file.write(path)

# Create test.txt file
test_file_path = os.path.join(output_path, 'test.txt')
with open(test_file_path, 'w') as test_file:
    for path in path_list_test:
        test_file.write(path)

# Initialize counter for classes
i = 0

# Create classes.names file from classes.txt
classes_names_path = os.path.join(output_path, "classes_names")
classes_txt_path = os.path.join(output_path, "classes.txt")
with open(classes_names_path, "w") as cls_file, open(classes_txt_path, "r") as text_file:
    for line in text_file:
        cls_file.write(line)
        i += 1

# Create image_data.data file
image_data_path = os.path.join(output_path, "image_data_data")
with open(image_data_path, "w") as data_file:
    data_file.write('classes=' + str(i) + '\n')
    data_file.write('train=' + train_file_path + '\n')
    data_file.write('valid=' + test_file_path + '\n')
    data_file.write('names=' + classes_names_path + '\n')
    data_file.write('backup=backup\n')

print("Files created successfully.")

      



