'''
This simulation executes a trained neural network that approximates the
closed form solution given by 2 axis inv kin
'''
from __future__ import division

import numpy as np
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


'''Global Variables'''
input_shape = 2
output_shape = 4
drop_rte = 0.1
hidden_neurons = [40, 40, 40, 40,output_shape]
model = FullyConnectedNetwork(input_shape, hidden_neurons, drop_rte)

def load_model(model):
    return model.load_state_dict(torch.load('./saved_models/deterministicmodel.pth'))
load_model(model)

if torch.cuda.is_available():
    model.cuda()
    print("Using GPU Acceleration")
else:
    print("Not Using GPU Acceleration")
model.eval() #Model has been previously been trained. Set model to eval mode to use all connections

red = (255, 0, 0)
white = (255, 255, 255)

pygame.init()
pygame.display.set_caption('Fully Connected Network Approximating Inverse Kinematics')
width = 1000
height = 1000
display = pygame.display.set_mode((width, height))
frame_clock = pygame.time.Clock()

origin = (width / 2.0, height / 2.0)

upperarm = ArmPart('./robot_arm/upperarm.png', scale=.8)
lowerarm = ArmPart('./robot_arm/lowerarm.png', scale=.9)

sprites = []
num_steps_0 = 0
num_steps_1 = 0
cur_radians_0 = 0
cur_radians_1 = 0
origin_1 = (0, 0)
rotate_rte_0 = 0
rotate_rte_1 = 0

'''Main Script Logic'''
while True:
    display.fill(white)

    #Check if mouse is pressed and add location to sprites list if it is pressed
    mouse_state = pygame.mouse.get_pressed()
    if mouse_state[0] == 1:
        sprites.append(pygame.mouse.get_pos())
        sprites = helpers.return_ordered(sprites)

    #If sprites list has elements and the rotation steps aren't equal to zero
    #Calculate Inv Kinematic solution for the most recent sprite in the sprites list
    if len(sprites) > 0 and num_steps_0 == 0 and num_steps_1 == 0:
        #prepare input for neural network
        if torch.cuda.is_available():
            input_to_model = Variable(torch.from_numpy(np.asarray([(sprites[0][0] - 500.0 + 420)/840, (sprites[0][1] - 500.0 + 420)/840])).float().cuda())
        else:
            input_to_model = Variable(torch.from_numpy(np.asarray([(sprites[0][0] - 500.0 + 420)/840, (sprites[0][1] - 500.0 + 420)/840])).float())

        #Inference model and calculate rotation steps needed
        with torch.no_grad():
            theta_0_sin, theta_0_cos, theta_1_sin, theta_1_cos = model.forward(input_to_model)

        theta_0 = np.arctan2(theta_0_sin.item(), theta_0_cos.item())
        theta_1 = np.arctan2(theta_1_sin.item(), theta_1_cos.item())
        theta_0, theta_1 = helpers.convert_normal_angle(theta_0, theta_1)

        if (sprites[0][0] >=0):
            theta_add = (theta_1 + theta_0)% (2 * np.pi)
        else:
            theta_add = (theta_1 - theta_0)% (2 * np.pi)

        num_steps_0, rotate_rte_0 = helpers.calc_rot(cur_radians_0, theta_0)
        num_steps_1, rotate_rte_1 = helpers.calc_rot(cur_radians_1, theta_add)

    #Rotate upper and lower arm
    if num_steps_0 > 0 and num_steps_1 == 0:
        ua_image, ua_rect = upperarm.rotate(rotate_rte_0)
        fa_image, fa_rect = lowerarm.rotate(0.0)
        num_steps_0 +=-1

    #Rotate lower arm
    elif num_steps_1 > 0 and num_steps_0 == 0:
        fa_image, fa_rect = lowerarm.rotate(rotate_rte_1)
        ua_image, ua_rect = upperarm.rotate(0.0)
        num_steps_1 += -1

    #Rotate upper arm
    elif num_steps_0 > 0 and num_steps_1 > 0:
        fa_image, fa_rect = lowerarm.rotate(rotate_rte_1)
        ua_image, ua_rect = upperarm.rotate(rotate_rte_0)
        num_steps_0 += -1
        num_steps_1 += -1

    #Arm has reached end point, pop sprite from sprites list
    if num_steps_1 == 0 and num_steps_0 == 0:
        fa_image, fa_rect = lowerarm.rotate(0.000)
        ua_image, ua_rect = upperarm.rotate(0.000)

        if len(sprites) > 0:
            sprites.pop(0)

    joints_x = np.cumsum([0,
                          upperarm.scale * np.cos(upperarm.rot_angle),
                          lowerarm.scale * np.cos(lowerarm.rot_angle)]) + origin[0]
    joints_y = np.cumsum([0,
                          upperarm.scale * np.sin(upperarm.rot_angle),
                          lowerarm.scale * np.sin(lowerarm.rot_angle)]) * -1 + origin[1]

    #Update location of joints
    joints = [(int(x), int(y)) for x,y in zip(joints_x, joints_y)]

    #Draw sprites on screen
    for sprite in sprites:
        pygame.draw.circle(display, red, sprite, 4)

    #Reposition upper arm and lower arm
    helpers.transform(ua_rect, joints[0], upperarm)
    helpers.transform(fa_rect, joints[1], lowerarm)

    #Draw upper and lower arm
    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)

    #Get current location of arm parts
    cur_radians_0 = helpers.print_angle(ua_rect.center[0], ua_rect.center[1], (500, 500))
    cur_radians_1 = helpers.print_angle(fa_rect.center[0], fa_rect.center[1], (joints[1][0], joints[1][1]))

    #Check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()

            sys.exit()

    pygame.display.update()
    frame_clock.tick(30)
