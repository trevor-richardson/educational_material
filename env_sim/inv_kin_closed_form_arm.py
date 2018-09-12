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
from robot_arm.arm_part import ArmPart
import helpers

'''Global Variables'''
black = (0, 0, 0)
gold = (255, 215, 0)
red = (255, 0, 0)
white = (255, 255, 255)
linkage_color = (128, 0, 0, 200) # fourth value specifies transparency
pygame.init()
pygame.display.set_caption('Inverse Kinematics Closed Form Solution')
width = 1000
height = 1000
display = pygame.display.set_mode((width, height))
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

mouse_bool = False
hand_offset = 35

'''Main Script Logic'''
while True:
    display.fill(white)
    mouse_state = pygame.mouse.get_pressed()

    if mouse_state[0] == 1:
        sprites.append(pygame.mouse.get_pos())
        sprites = helpers.return_ordered(sprites)

    if len(sprites) > 0 and num_steps_0 == 0 and num_steps_1 == 0:

        theta_0, theta_1 = helpers.inv_kin_2arm(sprites[0][0] - origin[0], sprites[0][1] - origin[1], upperarm.scale, lowerarm.scale - hand_offset) #error possible if width isnt the dimension of interest
        if theta_1 == -20 and theta_0 == -20:
            print("Impossible to move end effector to desired location")
            num_steps_0 = 0
            num_steps_1 = 0
        else:
            theta_0, theta_1 = helpers.convert_normal_angle(theta_0, theta_1)
            ''' Here is where I collected theta from before'''

            if (sprites[0][0] >=0):
                theta_add = (theta_1 + theta_0) % (2 * np.pi)
            else:
                theta_add = (theta_1 - theta_0) % (2 * np.pi)

            num_steps_0, rotate_rte_0 = helpers.calc_rot(cur_radians_0, theta_0)
            num_steps_1, rotate_rte_1 = helpers.calc_rot(cur_radians_1, theta_add)
            # print(num_steps_0, num_steps_1)

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

    for sprite in sprites:
        pygame.draw.circle(display, red, sprite, 4)

    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)

    cur_radians_0 = helpers.print_angle(ua_rect.center[0], ua_rect.center[1], (500, 500))
    cur_radians_1 = helpers.print_angle(fa_rect.center[0], fa_rect.center[1], (joints[1][0], joints[1][1]))

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()


    pygame.display.update()
    frame_clock.tick(30)
