"""
Checks if the files copied to the respective image directories and their annotations match up. Run this after running split_files.py

Usage: split_files.py [-h] [-p PARENT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -p, --parent_dir 
                        Path to the folder where image split files and annotations are stored
"""
import os
import pandas as pd
import numpy as np
import argparse

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Checks if the files copied to the respective image directories and their annotations match up. Run this after running split_files.py")
parser.add_argument("-p",
                    "--parent_dir",
                    help="Path to the folder where image split files and annotations are stored",
                    type=str)

args = parser.parse_args()


train_dir=os.path.join(args.parent_dir,"train_images")
test_dir=os.path.join(args.parent_dir,"test_images")
valid_dir=os.path.join(args.parent_dir,"validation_images")
image_dir=[train_dir,test_dir,valid_dir]

train_annot_dir=train_dir=os.path.join(args.parent_dir,"annotations","train.csv")
test_annot_dir=train_dir=os.path.join(args.parent_dir,"annotations","test.csv")
valid_annot_dir=train_dir=os.path.join(args.parent_dir,"annotations","validation.csv")
annot_dir=[train_annot_dir,test_annot_dir,valid_annot_dir]

error_flag=False
for index in range(3):
    image_files = [image for image in os.listdir(image_dir[index]) if image.endswith('.jpg')]
    df=pd.read_csv(annot_dir[index])
    file=np.array(df.filter(items=["filename"]))
    file_names=file + '.jpg'
    for image_name in image_files:
        if image_name not in file_names:
            print(f"Error: File {str(image_name)} not in {str(annot_dir[index])}")
            error_flag=True
if not error_flag:
    print("Split operation was successful!")