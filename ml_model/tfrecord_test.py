import tensorflow as tf
import shutil
import numpy as np
import os
import cv2

class TFRecordExtractor:
    def __init__(self, tfrecord_file):
        self.tfrecord_file = tfrecord_file

    def _extract_fn(self, tfrecord):
        # Extract the features using the keysets
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'rows': tf.FixedLenFeature([], tf.int64),
            'cols': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

        # Extract the data record
        sample = tf.parse_single_example(tfrecord, features)
        filename = sample['filename']
        #image = ''
        image = tf.image.decode_image(sample['image'])
        img_shape = tf.stack([sample['rows'], sample['cols'],
            sample['channels']])
        label = sample['label']

        return [image, label, filename, img_shape]

    def extract_image(self):
        print('In')
        # Create a folder to store the extracted images
        folder_path = './ExtractedImages'
        # Remove the folder tree
        shutil.rmtree(folder_path, ignore_errors=True)
        os.mkdir(folder_path)

        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #try:
                # Keep extracting data till tfrecord is empty
            count = 0
            #while True:
            for i in range(498):
                count += 1
                #print(count)
                image_data = sess.run(next_image_data)
                #print(image_data[2])

                if not np.array_equal(image_data[0].shape, image_data[3]):
                    print(
                    'Image {} not decoded properly'.format(image_data[2]))
                    continue
                print(image_data[2].decode('utf-8'))
                save_path = os.path.abspath(os.path.join(folder_path,
                    image_data[2].decode('utf-8')))
                print('Save path = ', save_path, ', Label = ',
                    image_data[1])
                cv2.imwrite(save_path, image_data[0])

            #except:
                #print('ERROR!')
                #pass

if __name__ == '__main__':
    t = TFRecordExtractor('images.tfrecord')
    t.extract_image()
