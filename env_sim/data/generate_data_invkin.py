import numpy as np
import argparse
import sys
import os

parser = argparse.ArgumentParser(description='Input for generating data points to be learned from')

parser.add_argument('--num_data_train', type=int, default=100000, metavar='N',
                    help='Input batch size for training (default: 100000)')

parser.add_argument('--num_data_test', type=int, default=50000, metavar='N',
                    help='Input batch size for training (default: 10000)')

args = parser.parse_args()

def save_data(data, name):
    dir_path = os.path.dirname(os.path.realpath('inv_kin_closed_form_arm.py'))
    np.save(dir_path + '/inv_kin_aprox/' + name, data)

def convert_lst_np(lst):
    arr = np.asarray(lst)
    print(arr.shape)
    return arr

def inv_kin_2arm(x, y, l0, l1):
    inside = (x**2 + y**2 - l0**2 - l1**2)/(2*l0*l1)
    inside = round(inside, 5)
    if (x**2 + y**2 )**.5 > l0 + l1 or abs(inside) > 1 or (x == 0 and y == 0):
        return -20, -20
    else:
        theta_1 = (np.arccos(inside))
        a = y * (l1 * np.cos(theta_1) + l0) - x * l1 * np.sin(theta_1)
        if x == 0:
            x = .00001
        b = x * (l1 * np.cos(theta_1) + l0) + y * l1 * np.sin(theta_1)

        if b == 0:
            return -20, -20
        theta_0 = np.arctan2(a, b)
    return theta_0, theta_1

def convert_normal_angle(t_0, t_1):
    if t_0 < 0:
        t_0 = 2* np.pi + t_0
    if t_1 < 0:
        t_1 = 2* np.pi + t_1

    return t_0, t_1

def main():
    '''Regression Data Generator'''
    lst = []
    larg_x = 1000
    larg_y = 1000
    hand_offset = 35
    length1 = 186
    length2 = 269
    length2 += -hand_offset
    radius = length2 + length1
    lower_bound = 0
    upper_bound = 1000

    for iterator in range(args.num_data_train):
        x = int(np.random.uniform(lower_bound, upper_bound))
        y = int(np.random.uniform(lower_bound, upper_bound))
        theta_0, theta_1 = inv_kin_2arm(x - 500.0, y - 500.0, length1, length2)
        if theta_0 == -20 and theta_1 == -20:
            f = 0
        else:
            theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)
            lst.append([x - 500.0, y - 500.0, np.sin(theta_0), np.cos(theta_0), np.sin(theta_1), np.cos(theta_1)])

    data = convert_lst_np(lst)
    save_data(data, "train")
    del(lst[:])

    for iterator in range(args.num_data_test):
        x = int(np.random.uniform(lower_bound, upper_bound))
        y = int(np.random.uniform(lower_bound, upper_bound))
        theta_0, theta_1 = inv_kin_2arm(x - 500.0, y - 500.0, length1, length2)
        if theta_0 == -20 and theta_1 == -20:
            f = 0
        else:
            theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)
            lst.append([x - 500.0, y - 500.0, np.sin(theta_0), np.cos(theta_0), np.sin(theta_1), np.cos(theta_1)])

    data = convert_lst_np(lst)
    save_data(data, "test")
    del(lst[:])

    '''Classification Data Generator'''
    # for iterator in range(args.num_data_train):
    #     x = int(np.random.uniform(lower_bound, upper_bound))
    #     y = int(np.random.uniform(lower_bound, upper_bound))
    #     theta_0, theta_1 = inv_kin_2arm(x - 500.0, y - 500.0, length1, length2)
    #     if theta_0 == -20 and theta_1 == -20:
    #         lst.append([[x - 500.0, y - 500.0], [0]])
    #     else:
    #         lst.append([[x - 500.0, y - 500.0], [1]])
    #
    # data = convert_lst_np(lst)
    # save_data(data, "train_classification")
    # del(lst[:])
    #
    # for iterator in range(args.num_data_test):
    #     x = int(np.random.uniform(lower_bound, upper_bound))
    #     y = int(np.random.uniform(lower_bound, upper_bound))
    #     theta_0, theta_1 = inv_kin_2arm(x - 500.0, y - 500.0, length1, length2)
    #     if theta_0 == -20 and theta_1 == -20:
    #         lst.append([[x - 500.0, y - 500.0], [0]])
    #     else:
    #         lst.append([[x - 500.0, y - 500.0], [1]])
    #
    # data = convert_lst_np(lst)
    # save_data(data, "test_classification")
    # del(lst[:])



if __name__ == '__main__':
    main()
