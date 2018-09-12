from __future__ import division

import numpy as np
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import pygame.locals

import numpy as np
import pygame
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from arm_part import ArmPart

black = (0, 0, 0)
gold = (255, 215, 0)
red = (255, 0, 0)
white = (255, 255, 255)
linkage_color = (128, 0, 0, 200) # fourth value specifies transparency

pygame.init()
pygame.display.set_caption('Learning and PCA')

width = 1000
height = 1000
display = pygame.display.set_mode((width, height))
frame_clock = pygame.time.Clock()

upperarm = ArmPart('upperarm.png', scale=.8)
lowerarm = ArmPart('lowerarm.png', scale=.9)

origin = (width / 2.0, height / 2.0)

num_steps_0 = 0
num_steps_1 = 0
cur_radians_0 = 0
cur_radians_1 = 0
origin_1 = (0, 0)
rotate_rte_0 = 0
rotate_rte_1 = 0


def calc_rot(rad_current, rad_desired):
    #this is how many radians I need to move in total
    desired_transform = rad_desired - rad_current
    oneeighty = 180/57.2958
    #This is to make sure the direction I am turning is the most efficient way to turn
    if desired_transform < 0:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = 1 * (1 / (57.2958)) #1 degree per frame
        else:
            rotation_rte = (-1 / (57.2958)) #1 degree per frame
    else:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = (-1/ (57.2958))
        else:
            rotation_rte = 1 * (1 / (57.2958)) #1 degree per frame

    #Number of steps moving at the specified rate
    desired_transform = (abs(desired_transform))
    if desired_transform > (np.pi):
        desired_transform = 2*np.pi - desired_transform

    num_steps = desired_transform / rotation_rte
    return int(abs(num_steps)), rotation_rte

def transform(rect, container, part):
    rect.center += np.asarray(container)
    rect.center += np.array([np.cos(part.rot_angle) * part.offset,
    -np.sin(part.rot_angle) * part.offset])

def print_angle(x, y, origin):
    if x <= origin[0] and y <= origin[1]:
        opposite = origin[1] - y
        adjacent = origin[0] - x
        if adjacent == 0.0:
            adjacent = .0001
        degree = np.arctan(opposite/adjacent) * 57.2958 + 180
    elif x <= origin[0] and y >= origin[1]:
        opposite = origin[0] - x
        adjacent = y - origin[1]
        if adjacent == 0:
            adjacent = .0001
        degree = np.arctan(opposite/adjacent) * 57.2958 + 90
    elif x >= origin[0] and y <= origin[1]:
        opposite = x - origin[0]
        adjacent = origin[1] - y
        if adjacent == 0:
            adjacent = .0001
        degree = np.arctan(opposite/adjacent) * 57.2958 + 270
    else:
        adjacent = x - origin[0]
        opposite = y - origin[1]
        if adjacent == 0:
            adjacent = .0001
        degree = np.arctan(opposite/adjacent) * 57.2958

    return (degree / 57.2958)

def inv_kin_2arm(x, y, l0, l1):
    inside = (x**2 + y**2 - l0**2 - l1**2)/(2*l0*l1)
    inside = round(inside, 5)
    if (x**2 + y**2 )**.5 > l0 + l1 or abs(inside) > 1 or (x == 0 and y == 0):
        print("returning -1 -1")
        return -1, -1
    else:
        theta_1 = (np.arccos(inside))
        a = y * (l1 * np.cos(theta_1) + l0) - x * l1 * np.sin(theta_1)
        if x == 0:
            x = .00001
        b = x * (l1 * np.cos(theta_1) + l0) + y * l1 * np.sin(theta_1)

        if b == 0:
            print("impossible to reach", l0, l1, x, y, abs(((x^2 + y^2) - l0^2 - l1^2)/(2*l0*l1)))
            return -1, -1
        theta_0 = np.arctan2(a, b)
    return theta_0, theta_1

def convert_normal_angle(t_0, t_1):
    if t_0 < 0:
        t_0 = 2* np.pi + t_0
    if t_1 < 0:
        t_1 = 2* np.pi + t_1
    return t_0, t_1

def rot_it(rad_current, rad_desired):
    #this is how many radians I need to move in total
    desired_transform = rad_desired - rad_current
    oneeighty = 180/57.2958

    #This is to make sure the direction I am turning is the most efficient way to turn
    if desired_transform < 0:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = 1 #1 degree per frame
        else:
            rotation_rte = -1 #1 degree per frame
    else:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = -1
        else:
            rotation_rte = 1 #1 degree per frame

    desired_transform = (abs(desired_transform))
    if desired_transform > (np.pi):
        desired_transform = 2*np.pi - desired_transform

    return 1, rotation_rte * desired_transform

def calculate_pca(data, k=2):
    #extract mean of the data
    data = torch.from_numpy(data)
    data_mean = torch.mean(data,0) #columnwise mean
    data = data - data_mean.expand_as(data)

    U,S,V = torch.svd(torch.t(data))
    return U[:,:k], data_mean[0], data_mean[1]

half_circle_bool = False
current_state = 0
trajectories_to_collect = 0 #This will collect one trajectory
joint_data = []
lamda = 0

basicfont = pygame.font.SysFont(None, 48)

while 1:
    display.fill(white)

    if current_state == 0 and num_steps_0 == 0 and num_steps_1 == 0 and cur_radians_0 != 0:
        if half_circle_bool:
            goal = (0, upperarm.scale + lowerarm.scale)
        else:
            goal = (0, upperarm.scale - lowerarm.scale)
        theta_0, theta_1 = inv_kin_2arm(goal[0], goal[1], upperarm.scale, lowerarm.scale)
        theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)

        if (goal[0] >=0):
            theta_add = (theta_1 + theta_0)% (2 * np.pi)
        else:
            theta_add = (theta_1 - theta_0)% (2 * np.pi)

        num_steps_0, rotate_rte_0 = calc_rot(cur_radians_0, theta_0)
        num_steps_1, rotate_rte_1 = calc_rot(cur_radians_1, theta_add)
    #current_state = 1 -- move arm to stretched out position
    if current_state == 1 and num_steps_0 == 0 and num_steps_1 == 0:
        goal = (upperarm.scale + lowerarm.scale, 0)
        trajectories_to_collect += -1

        theta_0, theta_1 = inv_kin_2arm(goal[0], goal[1], upperarm.scale, lowerarm.scale)
        theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)

        if (goal[0] >=0):
            theta_add = (theta_1 + theta_0)% (2 * np.pi)
        else:
            theta_add = (theta_1 - theta_0)% (2 * np.pi)

        num_steps_0, rotate_rte_0 = calc_rot(cur_radians_0, theta_0)
        num_steps_1, rotate_rte_1 = calc_rot(cur_radians_1, theta_add)
    #current_state = 2 -- move arm to starting position
    if current_state == 2 and num_steps_0 == 0 and num_steps_1 == 0:
        if half_circle_bool:
            goal = (0, -upperarm.scale - lowerarm.scale)
        else:
            goal = (0, upperarm.scale - lowerarm.scale)
        theta_0, theta_1 = inv_kin_2arm(goal[0], goal[1], upperarm.scale, lowerarm.scale)
        theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)

        if (goal[0] >=0):
            theta_add = (theta_1 + theta_0)% (2 * np.pi)
        else:
            theta_add = (theta_1 - theta_0)% (2 * np.pi)

        num_steps_0, rotate_rte_0 = calc_rot(cur_radians_0, theta_0)
        num_steps_1, rotate_rte_1 = calc_rot(cur_radians_1, theta_add)
    #current state = 3 -- train pca -- execute pca
    if current_state ==3 and num_steps_0 == 0 and num_steps_1 == 0:
        #calc PCA
        pc, m_0, m_1 = calculate_pca(np.asarray(joint_data))
        current_state = 4

    if current_state == 4:
        '''executing pca'''
        theta_0 = m_0 + lamda * pc[0][0]
        theta_add = m_1 + lamda * pc[0][1]

        num_steps_0, rotate_rte_0 = rot_it(cur_radians_0, theta_0)
        num_steps_1, rotate_rte_1 = rot_it(cur_radians_1, theta_add)
        lamda += .05



    if num_steps_0 > 0 and num_steps_1 == 0:
        ua_image, ua_rect = upperarm.rotate(rotate_rte_0)
        fa_image, fa_rect = lowerarm.rotate(0.0)
        num_steps_0 +=-1
    elif num_steps_1 > 0 and num_steps_0 == 0:
        fa_image, fa_rect = lowerarm.rotate(rotate_rte_1)
        ua_image, ua_rect = upperarm.rotate(0.0)
        num_steps_1 += -1
    elif num_steps_0 > 0 and num_steps_1 > 0:
        fa_image, fa_rect = lowerarm.rotate(rotate_rte_1)
        ua_image, ua_rect = upperarm.rotate(rotate_rte_0)
        num_steps_0 += -1
        num_steps_1 += -1
    else:
        fa_image, fa_rect = lowerarm.rotate(0.000)
        ua_image, ua_rect = upperarm.rotate(0.000)
        if current_state == 0 and cur_radians_0 != 0:
            current_state = 1
        elif current_state == 1:
            current_state = 2
        elif current_state == 2:
            if half_circle_bool:
                current_state = 3
            else:
                current_state = 1
                if trajectories_to_collect < 0:
                    current_state = 3

    if half_circle_bool:
        if current_state == 1 or current_state == 2:
            joint_data.append([cur_radians_0, cur_radians_1])
    else:
        if current_state == 1:
            joint_data.append([cur_radians_0, cur_radians_1])

    #moving the arms after the calculations are made
    joints_x = np.cumsum([0,
                          upperarm.scale * np.cos(upperarm.rot_angle),
                          lowerarm.scale * np.cos(lowerarm.rot_angle)]) + origin[0]
    joints_y = np.cumsum([0,
                          upperarm.scale * np.sin(upperarm.rot_angle),
                          lowerarm.scale * np.sin(lowerarm.rot_angle)]) * -1 + origin[1]

    joints = [(int(x), int(y)) for x,y in zip(joints_x, joints_y)]

    transform(ua_rect, joints[0], upperarm)
    transform(fa_rect, joints[1], lowerarm)

    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)

    cur_radians_0 = print_angle(ua_rect.center[0], ua_rect.center[1], (500, 500))

    cur_radians_1 = print_angle(fa_rect.center[0], fa_rect.center[1], (joints[1][0], joints[1][1]))

    if current_state == 0:
        f=0
    elif current_state != 4:
        text = basicfont.render('Collecting Data', True, (0, 0, 0), (255, 255, 255))
        textrect = text.get_rect()
        display.blit(text, textrect)
    else:
        text = basicfont.render("Executing PCA Lambda " + str(round(lamda, 1)), True, (0, 0, 0), (255, 255, 255))
        textrect = text.get_rect()
        display.blit(text, textrect)

    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()

    if lamda > 20 and current_state == 4:
        current_state = 0
        lamda = 0
        if half_circle_bool:
            half_circle_bool = False
        else:
            half_circle_bool = True
        del(joint_data[:])

    pygame.display.update()
    frame_clock.tick(30)
