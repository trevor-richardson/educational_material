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
red = (255, 0, 0)
white = (255, 255, 255)
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

hand_offset = 35


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
        theta_0, theta_1 = helpers.inv_kin_2arm(sprites[0][0] - origin[0], sprites[0][1] - origin[1], upperarm.scale, lowerarm.scale - hand_offset)
        #Check -20 -20 return from inv_kin_2arm function which means that the location is outside of the arms capability of reaching
        if theta_1 == -20 and theta_0 == -20:
            print("Impossible to move end effector to desired location")
            num_steps_0 = 0
            num_steps_1 = 0
        else:
            #Caclulate the desired rotation angles for the upper and lower arm
            theta_0, theta_1 = helpers.convert_normal_angle(theta_0, theta_1)
            if (sprites[0][0] >=0):
                theta_add = (theta_1 + theta_0) % (2 * np.pi)
            else:
                theta_add = (theta_1 - theta_0) % (2 * np.pi)
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

    #Update location of joints
    joints_x = np.cumsum([0,
                          upperarm.scale * np.cos(upperarm.rot_angle),
                          lowerarm.scale * np.cos(lowerarm.rot_angle)]) + origin[0]
    joints_y = np.cumsum([0,
                          upperarm.scale * np.sin(upperarm.rot_angle),
                          lowerarm.scale * np.sin(lowerarm.rot_angle)]) * -1 + origin[1]

    joints = [(int(x), int(y)) for x,y in zip(joints_x, joints_y)]

    #Reposition upper arm and lower arm
    helpers.transform(ua_rect, joints[0], upperarm)
    helpers.transform(fa_rect, joints[1], lowerarm)

    #Draw sprites on screen
    for sprite in sprites:
        pygame.draw.circle(display, red, sprite, 4)

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
