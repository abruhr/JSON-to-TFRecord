### **This Repository is used to split images and annotations into train, validation and test and generate Tensorflow Records for Image Object Detection model for the Hyper Kvasir segmented images dataset.**

The dataset can be downloaded here: https://osf.io/mh9sj/

1. split_files.py is used to create divisions of images and CSV (from JSON) into train, validation and test.

2. verification_script.py is used to check if the split of images match up with the respective CSVs.

3. image_augmentation.py is used to do the image augmentation for the training data. bbox_util.py and data_aug.py contain classes used for image augmentation. 

4. generate_tfrecord2.py is used to generate Tensorflow Records files.
