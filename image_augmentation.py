""" Used to augment training images for better variability in image and increase training data. Programmed to produce double the images but can skip images due augmentation errors.

usage: image_augmentation.py [-h] [-p PARENT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -p, --parent_dir 
                        Parent directory where the train .json and images files are stored.
  
"""
import os
import random 
import pandas as pd
import numpy as np
import cv2
import shutil
from data_aug import *
from bbox_util import *
import matplotlib.pyplot as plt
import argparse 

parser = argparse.ArgumentParser(
    description="Used to augment training images for better variability in image and increase training data. Programmed to produce double the images but can skip images due augmentation errors")
parser.add_argument("-p",
                    "--parent_dir",
                    help="Directory where image split files and annotations are stored",
                    type=str)

args = parser.parse_args()


image_dir=os.path.join(args.parent_dir,'train_images')
image_files = [image for image in os.listdir(image_dir) if image.endswith('.jpg')]

csv_dir = os.path.join(args.parent_dir,'annotations','train.csv')
image_augment_dir = os.path.join(args.parent_dir,'train_augment_images') 
verify_augment_dir = os.path.join(args.parent_dir,'check_augment_images')
annot_augment_dir = os.path.join(args.parent_dir,'annotations','train_augment.csv')

if os.path.exists(image_augment_dir):
    shutil.rmtree(image_augment_dir)
if os.path.exists(annot_augment_dir):
    os.remove(annot_augment_dir)
if os.path.exists(verify_augment_dir):
    shutil.rmtree(verify_augment_dir)



def return_bbox(df,filename):
    result_df = df.loc[df['filename'] == filename] 
    df=np.array(result_df.filter(items=['xmin','ymin','xmax','ymax']))
    expand_dim=np.zeros((len(df),1),np.uint8)
    bboxes=np.append(df,expand_dim,axis=1)
    bboxes=bboxes.astype(float)
    return bboxes

def write_csv(name,w,h,bbox_array):
    # print(bbox_array)
    filename=name
    width=w
    height=h
    class_label='polyp'
    xmin=bbox_array[0][0]
    ymin=bbox_array[0][1]
    xmax=bbox_array[0][2]
    ymax=bbox_array[0][3]
    value = (filename, width,height,class_label,xmin,ymin,xmax,ymax)
    return value
    

def image_augment(img,bboxes,img_w_shape,choice):
        if choice == 0:
            seq=Sequence([RandomHSV(30, 30, 10),
                        RandomHorizontalFlip(1),RandomRotate(random.randint(7,12))])
            img_1, bboxes_1 = seq(img.copy(), bboxes.copy())
            return 0, img_1, bboxes_1
        elif choice == 1:
            seq=Sequence([RandomHSV(20, 20, 10),
                        Rotate(90), Resize(random.randint(img_w_shape+50,img_w_shape+50))])
            img_1, bboxes_1 = seq(img.copy(), bboxes.copy())
            return 1, img_1, bboxes_1
        elif choice == 2:
            seq=Sequence([RandomHSV(10, 60, 5),
                        RandomHorizontalFlip(1), RandomShear(random.uniform(0.35,0.5))])
            img_1, bboxes_1 = seq(img.copy(), bboxes.copy())
            return 2, img_1, bboxes_1
        elif choice == 3:
            seq=Sequence([RandomHSV(10, 40, 30),RandomHorizontalFlip(), RandomScale(0.3), 
            RandomShear(0.3)])
            img_1, bboxes_1 = seq(img.copy(), bboxes.copy())
            return 3, img_1, bboxes_1

def main():
    shutil.copytree(image_dir, image_augment_dir)
    csv_list=[]
    i=0
    column_name = ['filename', 'width', 'height',
                    'class', 'xmin', 'ymin', 'xmax', 'ymax']
    original_df=pd.read_csv(csv_dir)
    for file in image_files:
        bbox = return_bbox(original_df,file)
        img = cv2.imread(os.path.join(image_dir,file))[:,:,::-1]
        img_h,img_w,_ = img.shape
        
        if not os.path.exists(image_augment_dir):
            os.makedirs(image_augment_dir)
        if not os.path.exists(verify_augment_dir):
            os.makedirs(verify_augment_dir)
        prev_choice = None
        # result_bbox = None
        for aug in range(2):
            skip_flag=False
            for iter in range(5):
                choice = random.randint(0,3)
                while choice == prev_choice:
                    choice = random.randint(0,3) # Two different augmentation choices per image
                try :
                    option, result_image, result_bbox = image_augment(img,bbox,img_w,choice)
                    break
                except IndexError:
                    if iter < 4:
                        print("Index Error has occurred. Try "+iter)
                    if iter == 4:
                        print("Due to index error skipping image: " + file)
                        skip_flag=True
            prev_choice=choice
            if skip_flag == False:
                outer_array,inner_array= result_bbox.shape # Check dimension of bounding box output
                if outer_array == 1 and inner_array ==5: # Discard images that dont have a bounding box on them after augmentation
                    filename = 'edit_' + str(option) + '_' + file
                    change_color = cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(image_augment_dir,filename),change_color)
                    
                    img_w_bbox = draw_rect(result_image, result_bbox)
                    change_color = cv2.cvtColor(img_w_bbox,cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(verify_augment_dir,filename),change_color)
                    value=write_csv(filename,img_w,img_h,result_bbox)
                    csv_list.append(value)
        i+=1
        print("Created image number: "+ str(i))
            
    augment_df = pd.DataFrame(csv_list, columns=column_name)
    augment_df.to_csv(annot_augment_dir, index=None)
    images=[original_df,augment_df]
    total_df=pd.concat(images)
    total_df.to_csv(annot_augment_dir, index=None)
    print('Augmented image CSV has been created!')
    print('Augmented training images has been created and original training images copied to ' + image_augment_dir)

main()