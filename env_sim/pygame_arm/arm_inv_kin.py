import numpy as np
import pygame
import pygame.locals

import numpy as np
import pygame
import sys
import math

class ArmPart:
    """
    A class for storing relevant arm segment information.
    """
    def __init__(self, pic, scale=1.0):
        self.base = pygame.image.load(pic)
        # some handy constants
        self.length = self.base.get_rect()[2]
        self.scale = self.length * scale
        self.offset = self.scale / 2.0

        self.rotation = 0.0 # in radians

    def rotate(self, rotation):
        """
        Rotates and re-centers the arm segment.
        """
        self.rotation += rotation
        # rotate our image
        image = pygame.transform.rotozoom(self.base, np.degrees(self.rotation), 1)
        # reset the center
        rect = image.get_rect()
        rect.center = (0, 0)

        return image, rect

black = (0, 0, 0)
red = (255, 0, 0)
white = (255, 255, 255)
arm_color = (50, 50, 50, 200) # fourth value specifies transparency

pygame.init()

width = 750
height = 750
display = pygame.display.set_mode((width, height))
fpsClock = pygame.time.Clock()

upperarm = ArmPart('upperarm.png', scale=.7)
forearm = ArmPart('forearm.png', scale=.8)

line_width = 15
line_upperarm = pygame.Surface((upperarm.scale, line_width), pygame.SRCALPHA, 32)
line_forearm = pygame.Surface((forearm.scale, line_width), pygame.SRCALPHA, 32)

line_upperarm.fill(arm_color)
line_forearm.fill(arm_color)

origin = (width / 2, height / 2)

def transform(rect, base, arm_part):
    rect.center += np.asarray(base)
    rect.center += np.array([np.cos(arm_part.rotation) * arm_part.offset,
                            -np.sin(arm_part.rotation) * arm_part.offset])

def transform_lines(rect, base, arm_part):
    transform(rect, base, arm_part)
    rect.center += np.array([-rect.width / 2.0, -rect.height / 2.0])

#send commands about the desired end angle and number of steps to get there


def calc_rot(rad_current, rad_desired):
    #this is how many radians I need to move in total
    desired_transform = rad_desired - rad_current
    #I want to move one degree per frame at a frame rate of 40
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

    #Number of steps
    desired_transform = (abs(desired_transform))
    if desired_transform > (np.pi):
        desired_transform = 2*np.pi - desired_transform

    num_steps = desired_transform / rotation_rte

    return int(abs(num_steps)), rotation_rte



def print_angle(x, y, origin):
    if x <= origin[0] and y <= origin[1]:
        opposite = origin[1] - y
        adjacent = origin[0] - x
        if adjacent == 0.0:
            adjacent = .0001
        degree = np.arctan(opposite/adjacent) * 57.2958 + 180
        # print(degree, "degrees 1", opposite, adjacent, opposite/adjacent)
    elif x <= origin[0] and y >= origin[1]:
        opposite = origin[0] - x
        adjacent = y - origin[1]
        if adjacent == 0:
            adjacent = .0001
        degree = np.arctan(opposite/adjacent) * 57.2958 + 90
        # print(degree, "degrees 2 ", opposite, adjacent, opposite/adjacent)
    elif x >= origin[0] and y <= origin[1]:
        opposite = x - origin[0]
        adjacent = origin[1] - y
        if adjacent == 0:
            adjacent = .0001
        degree = np.arctan(opposite/adjacent) * 57.2958 + 270
        # print(degree, "degrees 3 ", opposite, adjacent, opposite/adjacent)
    else:
        adjacent = x - origin[0]
        opposite = y - origin[1]
        if adjacent == 0:
            adjacent = .0001
        degree = np.arctan(opposite/adjacent) * 57.2958
        # print(degree, "degrees 4", opposite, adjacent, opposite/adjacent)

    return (degree / 57.2958)

#assumptions are that x and y have been reoriented to the coordinate space desired about the origin
#need to ask heni about these corner cases and this closed formed solution
#http://web.eecs.umich.edu/~ocj/courses/autorob/autorob_10_ik_closedform.pdf slide 43
def inv_kin_2arm(x, y, l0, l1):
    # print(x, y, "xy")
    inside = (x**2 + y**2 - l0**2 - l1**2)/(2*l0*l1)
    inside = round(inside, 5)

    if (x**2 + y**2 )**.5 > l0 + l1 or abs(inside) > 1 or x == 0 or y == 0:
        return -1, -1
    else:
        theta_1 = (math.acos(inside))

        a = y * (l1 * math.cos(theta_1) + l0) - x * l1 * math.sin(theta_1)
        b = x * (l1 * math.cos(theta_1) + l0) + y * l1 * math.sin(theta_1)

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

def calc_origin(theta, hyp):
    if theta < (np.pi/2.0):
        x = hyp * math.cos(theta)
        y = hyp * math.sin(theta)
        # print("one")
    elif theta < np.pi:
        theta = np.pi - theta
        x = -1 * (hyp * math.cos(theta))
        y = hyp * math.sin(theta)
    elif theta < (3/2.0) * np.pi:
        theta = (3/2.0) * np.pi - theta
        y = -1 * (hyp * math.cos(theta))
        x =  -1 * hyp * math.sin(theta)
    else:
        theta = 2 * np.pi - theta
        x = (hyp * math.cos(theta))
        y = -1 * hyp * math.sin(theta)

    return int(-y), int(-x)

def return_ordered(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

sprites = []
num_steps_0 = 0
num_steps_1 = 0
cur_radians_0 = 0
cur_radians_1 = 0
origin_1 = (0, 0)
rotate_rte_0 = 0
rotate_rte_1 = 0

mouse_bool = False
while 1:
    display.fill(white)
    mouse_state = pygame.mouse.get_pressed()

    #check if mouse is pressed
    if mouse_state[0] == 1:
        sprites.append(pygame.mouse.get_pos())
        sprites = return_ordered(sprites)

    #calc new inv kinematic angle
    if len(sprites) > 0 and num_steps_0 == 0 and num_steps_1 == 0:

        theta_0, theta_1 = inv_kin_2arm(sprites[0][0] - 375.0, sprites[0][1] - 375.0, 179, 149) #error possible if width isnt the dimension of interest

        if theta_1 == -1 and theta_0 == -1:
            print("Impossible to move end effector to desired location")
            num_steps_0 = 0
            num_steps_1 = 0
            sprites = sprites[:-1]
        else:
            theta_0, theta_1 = convert_normal_angle(theta_0, theta_1)

            if (sprites[-1][0] >=0):
                theta_add = (theta_1 + theta_0)% (2 * np.pi)
            else:
                theta_add = (theta_1 - theta_0)% (2 * np.pi)

            num_steps_0, rotate_rte_0 = calc_rot(cur_radians_0, theta_0)
            num_steps_1, rotate_rte_1 = calc_rot(cur_radians_1, theta_add)

    if num_steps_0 > 0 and num_steps_1 == 0:
        ua_image, ua_rect = upperarm.rotate(rotate_rte_0)
        fa_image, fa_rect = forearm.rotate(0.0)
        num_steps_0 +=-1
    elif num_steps_1 > 0 and num_steps_0 == 0:
        fa_image, fa_rect = forearm.rotate(rotate_rte_1)
        ua_image, ua_rect = upperarm.rotate(0.0)
        num_steps_1 += -1
    elif num_steps_0 > 0 and num_steps_1 > 0:
        fa_image, fa_rect = forearm.rotate(rotate_rte_1)
        ua_image, ua_rect = upperarm.rotate(rotate_rte_0)
        num_steps_0 += -1
        num_steps_1 += -1
    else:
        fa_image, fa_rect = forearm.rotate(0.000)
        ua_image, ua_rect = upperarm.rotate(0.000)
        if len(sprites) > 0:
            sprites.pop(0)

    #i didnt write this
    joints_x = np.cumsum([0,
                          upperarm.scale * np.cos(upperarm.rotation),
                          forearm.scale * np.cos(forearm.rotation)
                          ]) + origin[0]
    joints_y = np.cumsum([0,
                          upperarm.scale * np.sin(upperarm.rotation),
                          forearm.scale * np.sin(forearm.rotation)
                          ]) * -1 + origin[1]
    joints = [(int(x), int(y)) for x,y in zip(joints_x, joints_y)]

    transform(ua_rect, joints[0], upperarm)
    transform(fa_rect, joints[1], forearm)

    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)

    # rotate arm lines
    line_ua = pygame.transform.rotozoom(line_upperarm,
                                        np.degrees(upperarm.rotation), 1)
    line_fa = pygame.transform.rotozoom(line_forearm,
                                        np.degrees(forearm.rotation), 1)

    # translate arm lines
    lua_rect = line_ua.get_rect()
    transform_lines(lua_rect, joints[0], upperarm)

    lfa_rect = line_fa.get_rect()
    transform_lines(lfa_rect, joints[1], forearm)
    #didnt write this

    cur_radians_0 = print_angle(ua_rect.center[0], ua_rect.center[1], (375, 375))


    cur_radians_1 = print_angle(fa_rect.center[0], fa_rect.center[1], (joints[1][0], joints[1][1]))

    #i didnt write this
    display.blit(line_ua, lua_rect)
    display.blit(line_fa, lfa_rect)

    # draw circles at joints for pretty
    pygame.draw.circle(display, black, joints[0], 30)
    pygame.draw.circle(display, arm_color, joints[0], 12)
    pygame.draw.circle(display, black, joints[1], 20)
    pygame.draw.circle(display, arm_color, joints[1], 7)

    for sprite in sprites:
        pygame.draw.circle(display, red, sprite, 4)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(30)
