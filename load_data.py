# -*- coding: utf-8 -*-

"""
Load data from files based on pickle package
"""

import pandas
import os
import numpy as np
import pandas as pd
import parameters
import pickle  # amazing package!!!
import data_process


def display_info(x_train, y_train, x_test, y_test):
    print("x_train", end=" -> ")
    print(np.shape(x_train))
    print("y_train", end=" -> ")
    print(np.shape(y_train))
    print("x_test", end=" -> ")
    print(np.shape(x_test))
    print("y_test", end=" -> ")
    print(np.shape(y_test))


def get_data(file_path):
    data = pickle.load(open(file_path, 'rb'))
    return data


def detect_files(path):
    files = os.listdir(path)
    count = 0
    data_files = []
    for file in files:
        if file[0] != parameters.VALID_FILE_FLAG:
            continue
        count += 1
        data_files.append(os.path.join(path, file))
    return data_files


def load_train_test_data():
    all_data = []
    for i in range(len(parameters.FILE_LIST)):
        valid_files = detect_files(parameters.FILE_LIST[i])
        for j in range(len(valid_files)):
            all_data.append(get_data(valid_files[j]))
    all_data[0], all_data[2] = data_process.delete_zero_data(all_data[0]), data_process.delete_zero_data(all_data[2])
    all_data[0], all_data[2] = data_process.transfer_65536(all_data[0]), data_process.transfer_65536(all_data[2])
    all_data[1], all_data[3] = np.array(all_data[1]), np.array(all_data[3])
    return all_data[0], all_data[1], all_data[2], all_data[3]


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_train_test_data()
    display_info(x_train, y_train, x_test, y_test)
