""" Images Split and Convert to CSV Files for TF Model. Creates train, validation and test folders into the image directory.
Assumption: Single Bounding Box JSON file and images file to be split (split to train,test,validation)

usage: split_files.py [-h] [-j JSON_DIR] [-i IMAGE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -j JSON_DIR, --json_dir JSON_DIR
                        Path to the folder where the input .json files are stored.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as JSON_DIR.
"""
import os
from random import shuffle
import json
import pandas as pd
import argparse
import shutil

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Training, test, validation split for TF model.")
parser.add_argument("-j",
                    "--json_dir",
                    help="Path to the folder where the input json file is stored.",
                    type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. ",
                    type=str, default=None)

args = parser.parse_args()

# Initialize directories
parent_dir=os.path.join(args.image_dir,"..","CSVplusImagesSplit")
train_dir=os.path.join(parent_dir, "train_images")
valid_dir=os.path.join(parent_dir, "validation_images")
test_dir=os.path.join(parent_dir, "test_images")
bbox_annot=os.path.join(parent_dir, "annotations")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
if not os.path.exists(bbox_annot):
    os.makedirs(bbox_annot)

def get_file_list_from_dir(path_to_jpg):
    image_files = [pos_jpeg for pos_jpeg in os.listdir(path_to_jpg) if pos_jpeg.endswith('.jpg')]
    return image_files

def randomize_images(file_list):
    shuffled_list_to_return=list(file_list)
    shuffle(shuffled_list_to_return)
    return shuffled_list_to_return

def split_images(file_list,train_split):
    train_split_index = round(len(file_list) * train_split)
    test_split=round(((1-train_split)/2),1)
    test_split_index=round(len(file_list) * test_split)
    training = file_list[:train_split_index]
    validation_plus_test = file_list[train_split_index:]
    testing = validation_plus_test[:test_split_index]
    validation = validation_plus_test[test_split_index:]
    return training, validation, testing

def copy_images(train,valid,test):
    list_images=os.listdir(args.image_dir)
    for image in list_images:
        source=os.path.join(args.image_dir,image)
        dest_train=os.path.join(train_dir,image)
        dest_test=os.path.join(test_dir,image)
        dest_valid=os.path.join(valid_dir,image)
        if image in train:
            shutil.copy(source,dest_train)
        if image in test:
            shutil.copy(source,dest_test)
        if image in valid:
            shutil.copy(source,dest_valid)
    print("Images copied to destinations successfully!")

def json_split(train,valid,test):
    train_csv_list = []
    test_csv_list = []
    valid_csv_list = []
    data_file=open(args.json_dir)   
    data = json.load(data_file)
    column_name = ['filename', 'width', 'height',
                    'class', 'xmin', 'ymin', 'xmax', 'ymax']
    for key,value in data.items():
        key_name=key+'.jpg'
        if key_name in train:     
            filename_train=key
            width_train=value['width']
            height_train=value['height']
            bbox_list_train=value['bbox'][0]
            class_label_train=bbox_list_train['label']
            xmin_train=bbox_list_train['xmin']
            ymin_train=bbox_list_train['ymin']
            xmax_train=bbox_list_train['xmax']
            ymax_train=bbox_list_train['ymax']
            value_train = (filename_train, width_train,height_train,class_label_train,xmin_train,ymin_train,xmax_train,ymax_train)
            train_csv_list.append(value_train)
        
        if key_name in valid:
            filename_valid=key
            width_valid=value['width']
            height_valid=value['height']
            bbox_list_valid=value['bbox'][0]
            class_label_valid=bbox_list_valid['label']
            xmin_valid=bbox_list_valid['xmin']
            ymin_valid=bbox_list_valid['ymin']
            xmax_valid=bbox_list_valid['xmax']
            ymax_valid=bbox_list_valid['ymax']
            value_valid = (filename_valid, width_valid,height_valid,class_label_valid,xmin_valid,ymin_valid,xmax_valid,ymax_valid)
            valid_csv_list.append(value_valid)

        if key_name in test:
            filename_test=key
            width_test=value['width']
            height_test=value['height']
            bbox_list_test=value['bbox'][0]
            class_label_test=bbox_list_test['label']
            xmin_test=bbox_list_test['xmin']
            ymin_test=bbox_list_test['ymin']
            xmax_test=bbox_list_test['xmax']
            ymax_test=bbox_list_test['ymax']
            value_test = (filename_test, width_test,height_test,class_label_test,
            xmin_test,ymin_test,xmax_test,ymax_test)
            test_csv_list.append(value_test)

    train_csv_df = pd.DataFrame(train_csv_list, columns=column_name)
    valid_csv_df = pd.DataFrame(valid_csv_list, columns=column_name)
    test_csv_df = pd.DataFrame(test_csv_list, columns=column_name)
    
    train_csv_df.to_csv(os.path.join(bbox_annot,"train.csv"), index=None)
    valid_csv_df.to_csv(os.path.join(bbox_annot,"validation.csv"), index=None)
    test_csv_df.to_csv(os.path.join(bbox_annot,"test.csv"), index=None)
    print("Successfully created CSV Files!")
    # return csv_df

def main():
    image_files=get_file_list_from_dir(args.image_dir)
    shuffled_list=randomize_images(image_files)
    train, valid, test=split_images(shuffled_list,0.8)
    copy_images(train,valid,test)
    json_split(train,valid,test)
    # print(len(test), len(valid), len(train))

main()
