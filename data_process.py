# -*- coding: utf-8 -*-

"""
For preprocess of data:
1. get rid of abnormal data
2. normalization(maxmin or -μ/std)
"""

import numpy as np

import parameters


def is_all_zero(data_piece):
    for i in range(len(data_piece)):
        data_piece_piece = np.array(data_piece[i])
        for j in range(len(data_piece_piece)):
            if data_piece_piece[j] != 0:
                return False
    return True


def find_and_remove_zero(origin):
    result = []
    for i in range(len(origin)):
        if is_all_zero(origin[i]):
            continue
        else:
            result.append(np.array(origin[i]))
    return result


def delete_zero_data(data):
    for i in range(len(data)):
        data[i] = find_and_remove_zero(data[i])
    return data


def transfer_65536(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                for m in range(len(data[i][j][k])):
                    if data[i][j][k][m] > 32768:
                        data[i][j][k][m] -= 65536
    return data


def only_first_line(x_data):
    new_x = []
    for i in range(len(x_data)):
        new_piece = []
        for j in range(len(x_data[i])):
            data_t = x_data[i][j].T
            data_t_mean = []
            if len(data_t) != 5:
                print("Error! file:data_process.py line:59 m!=5")
            for m in range(len(data_t)):
                data_t_mean.append(np.mean(data_t[m][:]))
            new_piece.append(np.array(data_t_mean))
        new_x.append(np.array(new_piece))
    return new_x


def poly_fit(data_piece):
    data_piece = np.array(data_piece)
    data_t = data_piece.T
    x = [i for i in range(len(data_t[0]))]
    new_piece = np.zeros((parameters.WIDTH, parameters.HEIGHT))
    for i in range(len(data_t)):
        y = data_t[i][:]
        coe = np.polyfit(x, y, 5)
        poly = np.poly1d(coe)
        for j in range(parameters.WIDTH):
            x_value = j * (len(data_t[0])/parameters.WIDTH)
            new_piece[j][i] = poly(x_value)
    return new_piece


def reshape_width_height(x_data, y_data):
    new_x = []
    new_y = []
    for i in range(len(x_data)):
        if len(x_data[i]) < 5:
            continue
        new_x.append(poly_fit(x_data[i]))
        new_y.append(y_data[i])
    return new_x, new_y
