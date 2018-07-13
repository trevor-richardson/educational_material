'''
This simulation executes a trained neural network that approximates the
closed form solution given by 2 axis inv kin
'''
import numpy as np
import matplotlib
import matplotlib.backends.backend_agg as agg
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab
import warnings
warnings.filterwarnings("ignore")

import pygame
import pygame.locals

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from arm_part import ArmPart

'''Simple Regression FCN Model'''
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_neurons, dropout_rte):
        super(FullyConnectedNetwork, self).__init__()

        self.h_0 = nn.Linear(input_dim, num_hidden_neurons[0])
        self.h_1 = nn.Linear(num_hidden_neurons[0], num_hidden_neurons[1])
        self.h_2 = nn.Linear(num_hidden_neurons[1], num_hidden_neurons[2])
        self.h_3 = nn.Linear(num_hidden_neurons[2], num_hidden_neurons[3])
        self.h_4 = nn.Linear(num_hidden_neurons[3], num_hidden_neurons[4])

        self.drop = nn.Dropout(dropout_rte)

    def forward(self, x):

        out_0 = F.tanh(self.h_0(x))
        out_0 = self.drop(out_0)

        out_1 = F.tanh(self.h_1(out_0))
        out_1 = self.drop(out_1)

        out_2 = F.tanh(self.h_2(out_1))
        out_2 = self.drop(out_2)

        out_3 = F.tanh(self.h_3(out_2))

        out = self.h_4(out_3)
        return out

def load_model(model):
    # return model.load_state_dict(torch.load('/home/trevor/coding/educational_material/env_sim/pygame_arm/mysavedmodel.pth'))
    return model.load_state_dict(torch.load('/home/trevor/coding/educational_material/env_sim/pygame_arm/saved_models/deterministicmodel.pth'))

def make_uncertainty_plots(h, h_2, p, p2):
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
    fit_2 = stats.norm.pdf(h_2, np.mean(h_2), np.std(h_2))

    x = plt.figure(1)
    x.patch.set_facecolor('white')
    plt.subplot(211)
    plt.title("Theta 1")
    plt.plot(h,fit,'-o')
    plt.hist(h,normed=True)
    # plt.axvline(x=p, lw=4)
    plt.yticks([])
    plt.xlim((0,6.5))
    plt.xlabel("Radians")
    plt.subplot(212)
    plt.title("Theta 2")
    plt.plot(h_2,fit_2,'-o')
    plt.hist(h_2,normed=True)
    # plt.axvline(x=p2, lw=4)
    plt.yticks([])
    plt.xlim((0,6.5))
    plt.xlabel("Radians")
    plt.tight_layout()

    ax = plt.gca()

    canvas = agg.FigureCanvasAgg(x)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    size = canvas.get_width_height()
    plot_surface = pygame.image.fromstring(raw_data, size, "RGB")

    return plot_surface

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

upperarm = ArmPart('upperarm.png', scale=.8)
lowerarm = ArmPart('lowerarm.png', scale=.9)

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
save_data_bool = True
save_iterator = 2

lst_theta0 = []
lst_theta1 = []

def save_data(data, label, iteration):
    dir_path = os.path.dirname(os.path.realpath('inv_kin_closed_form_arm.py'))
    np.save(dir_path + '/data/data' + str(iteration), data)
    np.save(dir_path + '/data/label' + str(iteration), label)

def calc_rot(rad_current, rad_desired):
    #this is how many radians I need to move in total
    desired_transform = rad_desired - rad_current
    oneeighty = np.radians(180)
    #This is to make sure the direction I am turning is the most efficient way to turn
    if desired_transform < 0:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = 1 * np.radians(1) #1 degree per frame
        else:
            rotation_rte = -np.radians(1) #1 degree per frame
    else:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = -np.radians(1)
        else:
            rotation_rte = 1* np.radians(1) #1 degree per frame

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
        degree = np.degrees(np.arctan(opposite/adjacent)) + 180
    elif x <= origin[0] and y >= origin[1]:
        opposite = origin[0] - x
        adjacent = y - origin[1]
        if adjacent == 0:
            adjacent = .0001
        degree = np.degrees(np.arctan(opposite/adjacent)) + 90
    elif x >= origin[0] and y <= origin[1]:
        opposite = x - origin[0]
        adjacent = origin[1] - y
        if adjacent == 0:
            adjacent = .0001
        degree = np.degrees(np.arctan(opposite/adjacent)) + 270
    else:
        adjacent = x - origin[0]
        opposite = y - origin[1]
        if adjacent == 0:
            adjacent = .0001
        degree = np.degrees(np.arctan(opposite/adjacent))

    return np.radians(degree)


def convert_normal_angle(t_0, t_1):
    if t_0 < 0:
        t_0 = 2* np.pi + t_0
    if t_1 < 0:
        t_1 = 2* np.pi + t_1

    return t_0, t_1

def return_ordered(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

basicfont = pygame.font.SysFont(None, 38)

while 1:
    display.fill(white)
    mouse_state = pygame.mouse.get_pressed()

    if mouse_state[0] == 1:
        sprites.append(pygame.mouse.get_pos())
        sprites = return_ordered(sprites)

    if len(sprites) > 0 and num_steps_0 == 0 and num_steps_1 == 0 and mouse_state_bool:

        if torch.cuda.is_available():
            input_to_model = Variable(torch.from_numpy(np.asarray([(sprites[0][0] - origin[0] + 420)/840, (sprites[0][1] - origin[1] + 420)/840])).float().cuda())
        else:
            input_to_model = Variable(torch.from_numpy(np.asarray([(sprites[0][0] - origin[0] + 420)/840, (sprites[0][1] - origin[1] + 420)/840])).float())
        model.train()

        for iterator in range(sample_size_drop):
            theta_0_sin, theta_0_cos, theta_1_sin, theta_1_cos = model.forward(input_to_model)

            theta_0 = np.arctan2(theta_0_sin.data[0], theta_0_cos.data[0])
            theta_1 = np.arctan2(theta_1_sin.data[0], theta_1_cos.data[0])
            theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)
            lst_theta0.append(theta_0)
            lst_theta1.append(theta_1)
        optimizer.zero_grad() #dont want to store gradients

        if 'uncertainty_graphs' in locals():
            plt.clf()

        uncertainty_graphs = make_uncertainty_plots(sorted(lst_theta0[:-1]), sorted(lst_theta1[:-1]), lst_theta0[-1], lst_theta1[-1])
        del(lst_theta0[:])
        del(lst_theta1[:])

        model.eval()
        theta_0_sin, theta_0_cos, theta_1_sin, theta_1_cos = model.forward(input_to_model)

        theta_0 = np.arctan2(theta_0_sin.data[0], theta_0_cos.data[0])
        theta_1 = np.arctan2(theta_1_sin.data[0], theta_1_cos.data[0])
        theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)

        lst_theta0.append(theta_0)
        lst_theta1.append(theta_1)

        if (sprites[0][0] >=0):
            theta_add = (theta_1 + theta_0)% (2 * np.pi)
        else:
            theta_add = (theta_1 - theta_0)% (2 * np.pi)

        num_steps_0, rotate_rte_0 = calc_rot(cur_radians_0, theta_0)
        num_steps_1, rotate_rte_1 = calc_rot(cur_radians_1, theta_add)
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

    else:
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

    transform(ua_rect, joints[0], upperarm)
    transform(fa_rect, joints[1], lowerarm)

    if 'uncertainty_graphs' in locals():
        display.blit(uncertainty_graphs, (1020,250))

    for sprite in sprites:
        pygame.draw.circle(display, red, sprite, 4)

    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)

    cur_radians_0 = print_angle(ua_rect.center[0], ua_rect.center[1], (origin[0], origin[1]))
    cur_radians_1 = print_angle(fa_rect.center[0], fa_rect.center[1], (joints[1][0], joints[1][1]))

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
