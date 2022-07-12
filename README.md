### **This Repository is used to split images and annotations into train, validation and test and generate Tensorflow Records for Image Object Detection model.**

Assumption: JSON Annotation for all images is originally in one JSON file.

1. split_files.py is used to create divisions of images and CSV (from JSON) into train, validation and test.

2. verification_script.py is used to check if the split of images match up with the respective CSVs.

3. generate_tfrecord2.py is used to generate Tensorflow Records files.