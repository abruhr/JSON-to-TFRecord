""" CSV and Image to TFRecord converter.
Assumption: split_files.py has already been run and the divisions of train, validation and test CSV annotations and images has been created.

usage: generate_tfrecord.py [-h] [-p PARENT_DIR] [-o OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -p, --parent_dir 
                        Directory where image split files and annotations are stored
  -o, --use_augment 
                        Choose to use augmented train images folder. Default is None
  -p, --output_dir 
                        Directory for generating output TFRecord (.record) file.

"""

import os
# import glob
import pandas as pd
import io
import argparse
# import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow JSON-to-TFRecord converter")
parser.add_argument("-p",
                    "--parent_dir",
                    help="Directory where image split files and annotations are stored",
                    type=str)
parser.add_argument("-a",
                    "--use_augment",
                    help="Choose to use augmented train images folder. Default is None. Enter 'True' to use augmented images", type=bool, default=None)
parser.add_argument("-o",
                    "--output_dir",
                    help="Directory for output TFRecord (.record) file.", type=str)


args = parser.parse_args()

# if args.image_dir is None:
#     args.image_dir = args.json_dir

# label_map = label_map_util.load_labelmap(args.labels_path)
# label_map_dict = label_map_util.get_label_map_dict(label_map)

# Initialize the input and output directories into a list
if args.use_augment == True:
    train_dir=os.path.join(args.parent_dir,"train_augment_images")
else:
    train_dir=os.path.join(args.parent_dir,"train_images")

test_dir=os.path.join(args.parent_dir,"test_images")
valid_dir=os.path.join(args.parent_dir,"validation_images")
image_dir=[train_dir,test_dir,valid_dir]

if args.use_augment == True:
    train_annot_dir=os.path.join(args.parent_dir,"annotations","train_augment.csv")
else:
    train_annot_dir=os.path.join(args.parent_dir,"annotations","train.csv")
test_annot_dir=os.path.join(args.parent_dir,"annotations","test.csv")
valid_annot_dir=os.path.join(args.parent_dir,"annotations","validation.csv")
annot_dir=[train_annot_dir,test_annot_dir,valid_annot_dir]

train_record_dir=os.path.join(args.output_dir,"train.record")
test_record_dir=os.path.join(args.output_dir,"test.record")
valid_record_dir=os.path.join(args.output_dir,"validation.record")
tfrecord_dir=[train_record_dir,test_record_dir,valid_record_dir]


def load_csv(path):
    df=pd.read_csv(path)
    return df

# def json_to_csv(path):
#     csv_list = []
#     json_files=glob.glob(path + '/*.json')
#     for json_file in json_files:
#         data_file=open(json_file)   
#         data = json.load(data_file)
#         for key,value in data.items():
#             filename=key
#             width=value['width']
#             height=value['height']
#             bbox_list=value['bbox'][0]
#             class_label=bbox_list['label']
#             xmin=bbox_list['xmin']
#             ymin=bbox_list['ymin']
#             xmax=bbox_list['xmax']
#             ymax=bbox_list['ymax']
#             value = (filename, width,height,class_label,xmin,ymin,xmax,ymax)
#             csv_list.append(value)
#     column_name = ['filename', 'width', 'height',
#                     'class', 'xmin', 'ymin', 'xmax', 'ymax']
#     csv_df = pd.DataFrame(csv_list, columns=column_name)
#     return csv_df


def class_text_to_int(row_label):
    if row_label == 'polyp':
        return 1
    else:
        None



def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    for data in range(3):
        writer = tf.python_io.TFRecordWriter(tfrecord_dir[data])
        path = image_dir[data]
        examples = load_csv(annot_dir[data])
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecord file: {}'.format(str(tfrecord_dir[data])))

if __name__ == '__main__':
    tf.app.run()