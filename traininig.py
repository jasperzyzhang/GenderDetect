
import pandas as pd
import numpy as np
import cv2

import seaborn as sns
from sklearn.metrics import f1_score

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD

from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64





from generate import generate_df

def apply_mask(image, size=30, n_squares=5):
    h, w, channels = image.shape
    new_image = image
    for _ in range(n_squares):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)
        new_image[y1:y2,x1:x2,:] = 0
    return new_image



def train(path,imgpath,attr,weightname,train_num = 8000,val_num = 1000, epoch= 100 ,batch = 64):

    NUM_EPOCHS = epoch

    BATCH_SIZE = batch

    # Train data
    x_train, y_train = generate_df(0, 'Male', train_num,imgpath,attr)

    # Train - Data Preparation - Data Augmentation with generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=apply_mask,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_datagen.fit(x_train)

    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=BATCH_SIZE,
    )

    # Validation Data
    x_valid, y_valid = generate_df(1, 'Male', val_num,imgpath,attr)

    # Import InceptionV3 Model
    inc_model = InceptionV3(weights=path + 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            include_top=False,
                            input_shape=(218,178, 3))



    # Adding custom Layers
    x = inc_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    # creating the final model
    model_ = Model(inputs=inc_model.input, outputs=predictions)

    # Lock initial layers to do not be trained
    for layer in model_.layers[:52]:
        layer.trainable = False

    # compile the model
    model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9)
                   , loss='categorical_crossentropy'
                   , metrics=['accuracy'])

    # https://keras.io/models/sequential/ fit generator
    checkpointer = ModelCheckpoint(filepath= path + weightname + '.hdf5',
                                   verbose=1, save_best_only=True)

    hist = model_.fit_generator(train_generator
                                , validation_data=(x_valid, y_valid)
                                , steps_per_epoch= train_num / BATCH_SIZE
                                , epochs=NUM_EPOCHS
                                , callbacks=[checkpointer]
                                , verbose=1
                                )


    return hist,model_

# type == 2 : Data set used for training, type == 3: Google crawled data
def test(type,test_num,imgpath,attr,model_):

    # Test Data
    x_test, y_test = generate_df(type, 'Male', test_num,imgpath,attr)

    # generate prediction
    model_predictions = [np.argmax(model_.predict(feature)) for feature in x_test]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) == y_test) / len(model_predictions)
    print('Model Evaluation')
    print('Test accuracy: %.4f%%' % test_accuracy)
    f1 = f1_score(y_test, model_predictions)
    print('f1_score:', f1)

    return test_accuracy,f1