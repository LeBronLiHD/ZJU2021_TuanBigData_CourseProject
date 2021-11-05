# -*- coding: utf-8 -*-

"""
Train hand gesture recognization model using CNN algorithm
"""

import os
import random
import sys
import time
import load_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical
import parameters
import data_process


def get_activation():
    if parameters.CLASS_NUM == 2:
        return "sigmoid"
    else:
        return "softmax"


def CNN(X_train, Y_train, X_test, Y_test, train=True, ver=True):
    width, height = parameters.WIDTH, parameters.HEIGHT
    print("width =", width, "  height =", height)
    Y_test, Y_train = to_categorical(Y_test, num_classes=parameters.CLASS_NUM), \
                                to_categorical(Y_train, num_classes=parameters.CLASS_NUM)
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    load_data.display_info(X_train, Y_train, Y_test, Y_test)
    init_time = time.time()
    if train:
        TrainCnnModel(np.array(X_train), Y_train, width, height, np.array(X_test), Y_test)
    print("CNN done, time ->", time.time() - init_time)


def TrainCnnModel(x_train, y_train, width, height, x_test, y_test, big=False, exp=False):
    epoch_num = parameters.EPOCH_NUM
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(2, 2), padding='same',
                     input_shape=(width, height, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(parameters.CLASS_NUM, activation=get_activation()))
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=25, mode='max')
    history = model.fit(x_train, y_train, batch_size=16, epochs=epoch_num, verbose=1,
                        callbacks=[early_stopping], validation_data=(x_test, y_test), shuffle=True)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    history.history['accuracy'][0] = min(1.0, history.history['accuracy'][0])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    score = model.evaluate(x_test, y_test)
    print('model accuracy ->', score[1])
    # saving the model
    save_dir = parameters.MODEL_SAVE
    model_name = "model_cnn_" + str(epoch_num) + "_" + str(score[1]) + ".h5"
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('saved trained model at %s ' % model_path)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data.load_train_test_data()
    x_train, x_test = data_process.only_first_line(x_train), data_process.only_first_line(x_test)
    x_train, y_train = data_process.reshape_width_height(x_train, y_train)
    x_test, y_test = data_process.reshape_width_height(x_test, y_test)
    CNN(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test),
        train=True, ver=True)
