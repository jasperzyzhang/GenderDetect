
import pandas as pd
import numpy as np
import cv2
from os import listdir
import seaborn as sns

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils

from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64

IMG_WIDTH = 178
IMG_HEIGHT = 218



def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img) / 255.
    x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples,images_folder,df_par_attr):

    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
        3 -> test (with facemasks)
    '''

    df_ = df_par_attr[(df_par_attr['partition'] == partition)
                      & (df_par_attr[attr] == 0)].sample(int(num_samples / 2))
    df_ = pd.concat([df_,
                     df_par_attr[(df_par_attr['partition'] == partition)
                                 & (df_par_attr[attr] == 1)].sample(int(num_samples / 2))])

    # for Train and Validation
    if partition < 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr], 2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis=0)
            x_.append(im)
            y_.append(target[attr])
    return x_, y_