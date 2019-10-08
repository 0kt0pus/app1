import numpy as np
import tensorflow as tf
import os
#import matplotlib.image as mpimg
import cv2
import glob

class GenerateTFRecord:
    def __init__(self, labels):
        self.labels = labels
        print('hi')

    def convert_image_folder(self, img_folder, tfrecord_file_name):
        # get the file names of the images in the folder
        img_paths = os.walk(img_folder)
        #print(img_paths)
        filename_list = list()
        for i, name in enumerate(img_paths):
            if i != 0:
                path_name = name[0]
                for j in range(len(name[2])):
                    file_path = os.path.join(path_name, name[2][j])
                    filename_list.append(file_path)
        img_paths = [os.path.abspath(os.path.join(img_folder, i)) for i in filename_list]
        '''
        for img_path in img_paths:
            print(img_path)
            print('')
        '''
        with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
            for img_path in img_paths:
                example = self.convert_image(img_path)
                writer.write(example.SerializeToString())

    def convert_image(self, img_path):
        label = self.get_label_with_filename(img_path)
        #image_data = cv2.imread(img_path)
        #image_str = image_data.toString()
        image_shape = cv2.imread(img_path).shape
        filename = os.path.basename(img_path)
        # Read image data in terms of bytes
        with tf.gfile.FastGFile(img_path, 'rb') as fid:
            image_data = fid.read()

        # Write the tf record features
        example = tf.train.Example(features = tf.train.Features(feature = {
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(
                value = [filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(
                value = [image_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(
                value = [image_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(
                value = [image_shape[2]])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(
                value = [image_data])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(
                value = [label]))
        }))

        return example

    def get_label_with_filename(self, filename):
        basename = os.path.basename(filename).split('.')[0]
        basename = self.splitString(basename)
        return self.labels[basename]

    def splitString(self, str):

        alpha = ""
        num = ""
        special = ""
        for i in range(len(str)):
            if (str[i].isdigit()):
                num = num+ str[i]
            elif((str[i] >= 'A' and str[i] <= 'Z') or
                (str[i] >= 'a' and str[i] <= 'z')):
                alpha += str[i]
            else:
                special += str[i]

        return alpha

        #print(alpha)
        #print(num )
        #print(special)

if __name__ == '__main__':
    labels = {
        'Alfalfa': 0,
        'asparagus': 1,
        'Cattail': 2,
        'Teasel': 3,
        'Cleavers': 4,
        'Mallow': 5,
        'Elderberry': 6,
        'harebell': 7
    }
    t = GenerateTFRecord(labels)
    t.convert_image_folder(
    '/media/meluka/MELUKAS_BRAIN/app1/edible-wild-plants/datasets/dataset_train_v1',
    'images.tfrecord')
