import numpy as np
import pygame
import pygame.locals

import numpy as np
import pygame
import sys
import os

''' This class describes the png rect that I use to visualize in pygame '''
class ArmRect:
    def __init__(self, png, scale):
        self.contained = pygame.image.load(png)
        self.scale = self.contained.get_rect()[2] * scale
        self.offset = self.scale / 2.0
        self.rot_angle = 0.0

    def rotate(self, rotation):
        self.rot_angle += rotation
        image = pygame.transform.rotozoom(self.contained, np.degrees(self.rot_angle), 1)
        rect = image.get_rect()
        rect.center = (0, 0)

        return image, rect

black = (0, 0, 0)
gold = (255, 215, 0)
red = (255, 0, 0)
white = (255, 255, 255)
linkage_color = (128, 0, 0, 200) # fourth value specifies transparency

pygame.init()

width = 750
height = 750
display = pygame.display.set_mode((width, height))
frame_clock = pygame.time.Clock()

upperarm = ArmRect('upperarm.png', scale=.7)
lowerarm = ArmRect('lowerarm.png', scale=.8)

line_width = 12


line_upperarm = pygame.Surface((upperarm.scale, line_width), pygame.SRCALPHA, 32)
line_lowerarm = pygame.Surface((lowerarm.scale, line_width), pygame.SRCALPHA, 32)

line_upperarm.fill(linkage_color)
line_lowerarm.fill(linkage_color)

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

def transform(rect, container, part):
    rect.center += np.asarray(container)
    rect.center += np.array([np.cos(part.rot_angle) * part.offset,
                            -np.sin(part.rot_angle) * part.offset])

def transform_lines(rect, container, part):
    transform(rect, container, part)
    rect.center += np.array([-rect.width / 2.0, -rect.height / 2.0])

def calc_rot(rad_current, rad_desired):
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

#assumptions are that x and y have been reoriented to the coordinate space desired about the origin
#need to ask heni about these corner cases and this closed formed solution
#http://web.eecs.umich.edu/~ocj/courses/autorob/autorob_10_ik_closedform.pdf slide 43
def inv_kin_2arm(x, y, l0, l1):
    inside = (x**2 + y**2 - l0**2 - l1**2)/(2*l0*l1)
    inside = round(inside, 5)

    if (x**2 + y**2 )**.5 > l0 + l1 or abs(inside) > 1 or x == 0 or y == 0:
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
        theta_1, theta_0 = inv_kin_2arm(goal_pos[0] - 375, goal_pos[1] - 375, l0, l1)
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

current_pos_lst = [] #current end effector position
target_ja_lst = [] #traget_joint_angle

def save_data(data, name):
    dir_path = os.path.dirname(os.path.realpath('inv_kin_closed_form_arm.py'))
    np.save(dir_path + '/data/imitation_learning/' + name, data)

def create_dataset(current_pos, target_ja, goal_pos):
    data_lst = []
    for iterator in range(len(current_pos)):
        data_lst.append([goal_pos, current_pos[iterator], target_ja[iterator]])

    x = np.random.binomial(1, .8) #creates 0 or 1
    save_data(np.asarray(data_lst), str(np.random.randint(1000000)))

while 1:
    display.fill(white)
    mouse_state = pygame.mouse.get_pressed()

    #check if I have a current goal -- not generate goal -- true check if I have met goal position
    if goal_exists_bool:
        reached_goal = check_goal_status(current_endeff_pos, (goal_pos[0] -375, goal_pos[1]-375), 7) #seven represents the radius I accept as acceptable to goal point
        if reached_goal:
            create_dataset(current_pos_lst, target_ja_lst, (goal_pos[0] -375, goal_pos[1]-375))
            del(current_pos_lst)
            del(target_ja_lst)
            current_pos_lst = []
            target_ja_lst = []
            goal_exists_bool = False

    if not goal_exists_bool:
        goal_pos = generate_goal_pos(width, height, 179, 149)
        goal_exists_bool = True

    if mouse_state[0]  == 0:
        grabbed_endeff_bool=False

    if mouse_state[0] == 1 and not grabbed_endeff_bool:
        grabbed_endeff_bool = check_end_eff_pos((pygame.mouse.get_pos()[0]-375, pygame.mouse.get_pos()[1]-375 ), current_endeff_pos, 5)

    if mouse_state[0] == 1 and num_steps_0 == 0 and num_steps_1 == 0 and mouse_state_bool and grabbed_endeff_bool:
        sprites.append(pygame.mouse.get_pos())
        sprites = return_ordered(sprites)

        theta_0, theta_1 = inv_kin_2arm(sprites[0][0] - 375.0, sprites[0][1] - 375.0, 179, 149) #error possible if width isnt the dimension of interest
        if theta_1 == -20 and theta_0 == -20:
            print("Impossible to move end effector to desired location")
            num_steps_0 = 0
            num_steps_1 = 0
        else:
            theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)
            ''' Here is where I collected theta from before'''
            if grabbed_endeff_bool:
                current_pos_lst.append(current_endeff_pos)
                target_ja_lst.append([theta_0, theta_1])

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

    current_endeff_pos = (joints[2][0]-375, joints[2][1]-375)

    transform(ua_rect, joints[0], upperarm)
    transform(fa_rect, joints[1], lowerarm)

    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)

    # rotate arm lines
    line_ua = pygame.transform.rotozoom(line_upperarm,
                                        np.degrees(upperarm.rot_angle), 1)
    line_fa = pygame.transform.rotozoom(line_lowerarm,
                                        np.degrees(lowerarm.rot_angle), 1)

    # translate arm lines
    lua_rect = line_ua.get_rect()
    transform_lines(lua_rect, joints[0], upperarm)

    lfa_rect = line_fa.get_rect()
    transform_lines(lfa_rect, joints[1], lowerarm)

    cur_radians_0 = print_angle(ua_rect.center[0], ua_rect.center[1], (375, 375))

    cur_radians_1 = print_angle(fa_rect.center[0], fa_rect.center[1], (joints[1][0], joints[1][1]))

    display.blit(line_ua, lua_rect)
    display.blit(line_fa, lfa_rect)

    # make my joints more pronounced
    pygame.draw.circle(display, black, joints[0], 24)
    pygame.draw.circle(display, gold, joints[0], 12)
    pygame.draw.circle(display, black, joints[1], 16)
    pygame.draw.circle(display, gold, joints[1], 7)

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
