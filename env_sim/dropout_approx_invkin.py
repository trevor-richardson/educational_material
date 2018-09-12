'''
This simulation executes a trained neural network that approximates the
closed form solution given by 2 axis inv kin
'''
from __future__ import division
import numpy as np
import matplotlib
import matplotlib.backends.backend_agg as agg
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab
import warnings
warnings.filterwarnings("ignore")
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import pygame.locals
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from robot_arm.arm_part import ArmPart
import helpers
from neural_network import FullyConnectedNetwork

'''Simple Regression FCN Model'''
'''Define and Initialize Model'''


def load_model(model):
    return model.load_state_dict(torch.load('./saved_models/deterministicmodel.pth'))

learning_rate = .0001
sample_size_drop = 60
input_shape = 2
output_shape = 4
drop_rte = 0.1
hidden_neurons = [40, 40, 40, 40,output_shape]
model = FullyConnectedNetwork(input_shape, hidden_neurons, drop_rte)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

load_model(model)
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU Acceleration")
else:
    print("Not Using GPU Acceleration")


'''Pygame Parameters'''
black = (0, 0, 0)
gold = (255, 215, 0)
red = (255, 0, 0)
white = (255, 255, 255)
linkage_color = (128, 0, 0, 200) # fourth value specifies transparency

pygame.init()
pygame.display.set_caption('Estimating Neural Network Confidence Using Dropout')

width = 1000
height = 1000
distance_for_histogram = 700
display = pygame.display.set_mode((width + distance_for_histogram, height))
frame_clock = pygame.time.Clock()

upperarm = ArmPart('./robot_arm/upperarm.png', scale=.8)
lowerarm = ArmPart('./robot_arm/lowerarm.png', scale=.9)

origin = (width / 2.0, height / 2.0)

sprites = []
num_steps_0 = 0
num_steps_1 = 0
cur_radians_0 = 0
cur_radians_1 = 0
origin_1 = (0, 0)
rotate_rte_0 = 0
rotate_rte_1 = 0

mouse_state_bool = True
save_iterator = 2

lst_theta0 = []
lst_theta1 = []

basicfont = pygame.font.SysFont(None, 38)


'''Main Script Logic'''
while 1:
    display.fill(white)
    mouse_state = pygame.mouse.get_pressed()

    if mouse_state[0] == 1:
        sprites.append(pygame.mouse.get_pos())
        sprites = helpers.return_ordered(sprites)

    if len(sprites) > 0 and num_steps_0 == 0 and num_steps_1 == 0 and mouse_state_bool:

        if torch.cuda.is_available():
            input_to_model = Variable(torch.from_numpy(np.asarray([(sprites[0][0] - origin[0] + 420)/840, (sprites[0][1] - origin[1] + 420)/840])).float().cuda())
        else:
            input_to_model = Variable(torch.from_numpy(np.asarray([(sprites[0][0] - origin[0] + 420)/840, (sprites[0][1] - origin[1] + 420)/840])).float())
        model.train()

        with torch.no_grad():
            for iterator in range(sample_size_drop):
                theta_0_sin, theta_0_cos, theta_1_sin, theta_1_cos = model.forward(input_to_model)

                theta_0 = np.arctan2(theta_0_sin.item(), theta_0_cos.item())
                theta_1 = np.arctan2(theta_1_sin.item(), theta_1_cos.item())
                theta_0, theta_1 = helpers.convert_normal_angle(theta_0, theta_1)
                lst_theta0.append(theta_0)
                lst_theta1.append(theta_1)

        if 'uncertainty_graphs' in locals():
            plt.clf()

        uncertainty_graphs = helpers.make_uncertainty_plots(sorted(lst_theta0[:-1]), sorted(lst_theta1[:-1]), lst_theta0[-1], lst_theta1[-1])
        del(lst_theta0[:])
        del(lst_theta1[:])

        model.eval()
        theta_0_sin, theta_0_cos, theta_1_sin, theta_1_cos = model.forward(input_to_model)

        theta_0 = np.arctan2(theta_0_sin.data[0], theta_0_cos.data[0])
        theta_1 = np.arctan2(theta_1_sin.data[0], theta_1_cos.data[0])
        theta_0, theta_1 = helpers.convert_normal_angle(theta_0, theta_1)

        lst_theta0.append(theta_0)
        lst_theta1.append(theta_1)

        if (sprites[0][0] >=0):
            theta_add = (theta_1 + theta_0)% (2 * np.pi)
        else:
            theta_add = (theta_1 - theta_0)% (2 * np.pi)

        num_steps_0, rotate_rte_0 = helpers.calc_rot(cur_radians_0, theta_0)
        num_steps_1, rotate_rte_1 = helpers.calc_rot(cur_radians_1, theta_add)
        mouse_state_bool = False

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

    if num_steps_1 == 0 and num_steps_0 == 0: #first edit in order to stabalize trajectory for python2 support
        fa_image, fa_rect = lowerarm.rotate(0.000)
        ua_image, ua_rect = upperarm.rotate(0.000)
        mouse_state_bool = True

        if len(sprites) > 0:
            sprites.pop(0)

    joints_x = np.cumsum([0,
                          upperarm.scale * np.cos(upperarm.rot_angle),
                          lowerarm.scale * np.cos(lowerarm.rot_angle)]) + origin[0]
    joints_y = np.cumsum([0,
                          upperarm.scale * np.sin(upperarm.rot_angle),
                          lowerarm.scale * np.sin(lowerarm.rot_angle)]) * -1 + origin[1]

    joints = [(int(x), int(y)) for x,y in zip(joints_x, joints_y)]

    helpers.transform(ua_rect, joints[0], upperarm)
    helpers.transform(fa_rect, joints[1], lowerarm)

    if 'uncertainty_graphs' in locals():
        display.blit(uncertainty_graphs, (1020,250))

    for sprite in sprites:
        pygame.draw.circle(display, red, sprite, 4)

    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)

    cur_radians_0 = helpers.print_angle(ua_rect.center[0], ua_rect.center[1], (origin[0], origin[1]))
    cur_radians_1 = helpers.print_angle(fa_rect.center[0], fa_rect.center[1], (joints[1][0], joints[1][1]))

    '''View text above my graphs'''
    text = basicfont.render('Model Uncertainty Graph', True, (0, 0, 0), (255, 255, 255))
    textrect = text.get_rect()
    textrect[0]+= 1195
    textrect[1]+=185
    display.blit(text, textrect)
    text2 = basicfont.render('60 Stochastic Forward Passes', True, (0, 0, 0), (255, 255, 255))
    textrect2 = text2.get_rect()
    textrect2[0]+= 1165
    textrect2[1]+=225
    display.blit(text2, textrect2)

    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()

            sys.exit()

    pygame.display.update()
    frame_clock.tick(30)
