import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

class Dataset_cap():
    def __init__(self, img_folder, char_to_num):
        self.img_folder = img_folder
        self.char_to_num = char_to_num

    def __encode_single_sample__(self, img_path, label, crop=False):
        # first need to read decode (to grayscale) images
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        # make tensors float32
        img = tf.image.convert_image_dtype(img, tf.float32)
        # crop image or not?
        if (crop == True):
            img = tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=25, target_height=50,
                                                target_width=125)
            img = tf.image.resize(img, size=[50, 200], method='bilinear', preserve_aspect_ratio=False, antialias=False,
                                  name=None)
        img = tf.transpose(img, perm=[1, 0, 2])
        # Convert strings to codes and make labels to train
        label = list(map(lambda x: self.char_to_num[x], label))
        return img.numpy(), label

    def data(self, crop=False):
        # Loop on all the files to create X whose shape is (1070, 50, 200, 1) and y whose shape is (1070, 5)
        X, y = [], []
        for _, _, files in os.walk(self.img_folder):
            for f in files:
                # To start, let's ignore the jpg images
                label = f.split('.')[0]
                extension = f.split('.')[1]
                if extension == 'png' or 'jpg':
                    img, label = self.__encode_single_sample__(self.img_folder + f, label, crop)
                    X.append(img)
                    y.append(label)
        X = np.array(X)
        y = np.array(y)

        # Split X, y to get X_train, y_train, X_val, y_val
        X_train, X_val, y_train, y_val = train_test_split(X.reshape(1070, 10000), y, test_size=0.1, shuffle=True,
                                                          random_state=42)
        X_train, X_val = X_train.reshape(963, 200, 50, 1), X_val.reshape(107, 200, 50, 1)
        return X_train, X_val, y_train, y_val