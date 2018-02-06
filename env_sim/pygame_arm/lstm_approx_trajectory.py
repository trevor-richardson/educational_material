import numpy as np
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

'''Model class'''
class ClassicalLSTM(nn.Module):
    def __init__(self, input_sz, hidden0_sz, hidden1_sz, hidden2_sz, output_sz):
        super(ClassicalLSTM, self).__init__()
        self.lstm_0 = nn.LSTMCell(input_sz, hidden0_sz)
        self.lstm_1 = nn.LSTMCell(hidden0_sz, hidden1_sz)
        self.lstm_2 = nn.LSTMCell(hidden1_sz, hidden2_sz)
        self.fcn1 = nn.Linear(hidden2_sz, output_sz)

    def forward(self, x, states):
        hx_0, cx_0 = self.lstm_0(x, states[0])
        hx_1, cx_1 = self.lstm_1(hx_0, states[1])
        hx_2, cx_2 = self.lstm_2(hx_1, states[2])
        x = self.fcn1(hx_2)
        return x, [[hx_0, cx_0], [hx_1, cx_1], [hx_2, cx_2]]

def load_model(model):
    return model.load_state_dict(torch.load('/home/trevor/coding/educational_material/env_sim/pygame_arm/lstmsavedmodel.pth'))

input_sz = 4
hidden0_sz = 40
hidden1_sz = 30
hidden2_sz = 15
output_sz = 4
learning_rate = .0001

model = ClassicalLSTM(input_sz, hidden0_sz, hidden1_sz, hidden2_sz, output_sz)
load_model(model)

if torch.cuda.is_available():
    model.cuda()
    print("Using GPU acceleration")
else:
    print("No GPU acceleration")

black = (0, 0, 0)
gold = (255, 215, 0)
red = (255, 0, 0)
white = (255, 255, 255)
linkage_color = (128, 0, 0, 200) # fourth value specifies transparency

pygame.init()

width = 1000
height = 1000
display = pygame.display.set_mode((width, height))
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

mouse_bool = False
mouse_state_bool = True
grabbed_endeff_bool = False #This only becomes True when the user grabs the location near the endeff
goal_exists_bool = False #Is there a current goal
reached_goal = False

current_pos_lst = [] #current end effector position
target_ja_lst = [] #traget_joint_angle

def create_lstm_states(n, batch_size):
    if torch.cuda.is_available():
        c = Variable(torch.zeros(batch_size, n)).float().cuda()
        h = Variable(torch.zeros(batch_size, n)).float().cuda()
    else:
        c = Variable(torch.zeros(batch_size, n)).float()
        h = Variable(torch.zeros(batch_size, n)).float()
    return (h, c)

def transform(rect, container, part):
    rect.center += np.asarray(container)
    rect.center += np.array([np.cos(part.rot_angle) * part.offset,
                            -np.sin(part.rot_angle) * part.offset])


def calc_rot(rad_current, rad_desired):
    desired_transform = rad_desired - rad_current
    oneeighty = 180/57.2958

    #This is to make sure the direction I am turning is the most efficient way to turn
    if desired_transform < 0:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = 1 #1 degree per frames
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
        return -20, -20
    else:
        theta_1 = (np.arccos(inside))
        a = y * (l1 * np.cos(theta_1) + l0) - x * l1 * np.sin(theta_1)
        b = x * (l1 * np.cos(theta_1) + l0) + y * l1 * np.sin(theta_1)
        if b == 0:
            print("impossible to reach")
            return -20, -20

        theta_0 = np.arctan2(a, b)

    return theta_0, theta_1

def convert_normal_angle(t_0, t_1):
    if t_0 < 0:
        t_0 = 2* np.pi + t_0
    if t_1 < 0:
        t_1 = 2* np.pi + t_1

    return t_0, t_1

def calc_origin(theta, hyp):
    if theta < (np.pi/2.0):
        x = hyp * np.cos(theta)
        y = hyp * np.sin(theta)
    elif theta < np.pi:
        theta = np.pi - theta
        x = -1 * (hyp * np.cos(theta))
        y = hyp * np.sin(theta)
    elif theta < (3/2.0) * np.pi:
        theta = (3/2.0) * np.pi - theta
        y = -1 * (hyp * np.cos(theta))
        x =  -1 * hyp * np.sin(theta)
    else:
        theta = 2 * np.pi - theta
        x = (hyp * np.cos(theta))
        y = -1 * hyp * np.sin(theta)

    return int(-y), int(-x)

''' Generate New Desired Location '''
def generate_goal_pos(width, height, l0, l1):
    feasible_solution = False
    while not feasible_solution:
        goal_pos = (int(np.random.uniform(0, width)), int(np.random.uniform(0, height)))
        theta_1, theta_0 = inv_kin_2arm(goal_pos[0] - 500, goal_pos[1] - 500, l0, l1)
        if theta_1 == -20 and theta_0 == -20:
            feasible_solution = False
        else:
            feasible_solution = True

    return goal_pos

''' Check if in goal circle '''
def check_goal_status(eff_pos, goal_pos, radius):
    if ((eff_pos[0] - goal_pos[0])**2 + (eff_pos[1] - goal_pos[1])**2 < radius**2 ):
        return True
    else:
        return False

''' Check if I grabbed end effector '''
def check_end_eff_pos(mouse_pos, eff_pos, radius):
    if ((eff_pos[0] - mouse_pos[0])**2 + (eff_pos[1] - mouse_pos[1])**2 < radius**2):
        return True
    else:
        return False

def return_ordered(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def save_data(data, name):
    dir_path = os.path.dirname(os.path.realpath('inv_kin_closed_form_arm.py'))
    np.save(dir_path + '/data/imitation_learning/' + name, data)

model.eval()
while 1:
    display.fill(white)

    #check if I have a current goal -- not generate goal -- true check if I have met goal position
    if goal_exists_bool:
        #check if I've met goal criteria
        reached_goal = check_goal_status(current_endeff_pos, (goal_pos[0] -500, goal_pos[1]-500), 10) #seven represents the radius I accept as acceptable to goal point
        if reached_goal:
            goal_exists_bool = False
        else:
            #prepare data for model
            input_data = np.asarray([(goal_pos[0] -500 + 420) / 840, (goal_pos[1] -500 + 420) / 840, (current_endeff_pos[0] -500 + 420) / 840, (current_endeff_pos[1] -500 + 420) / 840])
            if torch.cuda.is_available():
                data = Variable(torch.from_numpy(input_data).float().cuda(), volatile=True)
            else:
                data = Variable(torch.from_numpy(input_data).float(), volatile=True)

            output, states = model.forward(data, states)

            theta_0_sin = output.data[0][0]
            theta_0_cos = output.data[0][1]
            theta_1_sin = output.data[0][2]
            theta_1_cos = output.data[0][3]

            theta_0 = np.arctan2(theta_0_sin, theta_0_cos)
            theta_1 = np.arctan2(theta_1_sin, theta_1_cos)
            theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)

            if (current_endeff_pos[0] >=0):
                theta_add = (theta_1 + theta_0)% (2 * np.pi)
            else:
                theta_add = (theta_1 - theta_0)% (2 * np.pi)

            num_steps_0, rotate_rte_0 = calc_rot(cur_radians_0, theta_0)
            num_steps_1, rotate_rte_1 = calc_rot(cur_radians_1, theta_add)

    if not goal_exists_bool:
        #Here is where I need to reset states
        goal_pos = generate_goal_pos(width, height, 179, 149)
        goal_exists_bool = True
        prev0 = create_lstm_states(hidden0_sz, 1)
        prev1 = create_lstm_states(hidden1_sz, 1)
        prev2 = create_lstm_states(hidden2_sz, 1)
        states = [prev0, prev1, prev2]

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
        if len(sprites) > 0:
            sprites.pop(0)

    joints_x = np.cumsum([0,
                          upperarm.scale * np.cos(upperarm.rot_angle),
                          lowerarm.scale * np.cos(lowerarm.rot_angle)]) + origin[0]
    joints_y = np.cumsum([0,
                          upperarm.scale * np.sin(upperarm.rot_angle),
                          lowerarm.scale * np.sin(lowerarm.rot_angle)]) * -1 + origin[1]

    joints = [(int(x), int(y)) for x,y in zip(joints_x, joints_y)]

    current_endeff_pos = (joints[2][0]-500, joints[2][1]-500)

    transform(ua_rect, joints[0], upperarm)
    transform(fa_rect, joints[1], lowerarm)

    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)




    cur_radians_0 = print_angle(ua_rect.center[0], ua_rect.center[1], (500, 500))

    cur_radians_1 = print_angle(fa_rect.center[0], fa_rect.center[1], (joints[1][0], joints[1][1]))




    # draw circle at goal position
    pygame.draw.circle(display, red, goal_pos, 10)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:

            pygame.quit()
            sys.exit()

    pygame.display.update()
    frame_clock.tick(30)
    first_pass = False
