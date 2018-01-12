import numpy as np
import pygame
import pygame.locals
import sys

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

black = (0, 0, 255)
white = (255, 255, 255)

pygame.init()

width = 500
height = 500


display = pygame.display.set_mode((width, height))
fpsClock = pygame.time.Clock()

upperarm = ArmPart('forearm.png', scale=1)

base = (int(width / 2), int(height / 2))
sprites = [] #locations of the mouse

#These are the two booleans that allow one to check if a user held the mouse and let it go
only_one_hold = False #Makes sure I only count one mouse hold
mouse_hold_done = False #Makes sure I wait until user lets go of the mouse
previous_hold = False #Makes sure that the previous state was true

def print_angle(x, y):
    if x <= 250 and y <= 250:
        opposite = 250.0 - x
        adjacent = 250.0 - y
        if adjacent == 0.0:
            adjacent = .01
        degree = np.arctan(opposite/adjacent) * 57.2958
        # print(degree, "degrees 1", opposite, adjacent, opposite/adjacent)
    elif x <= 250 and y >= 250:
        opposite = y - 250.0
        adjacent = 250.0 - x
        if adjacent == 0:
            adjacent = .01
        degree = np.arctan(opposite/adjacent) * 57.2958 + 90
        # print(degree, "degrees 2 ", opposite, adjacent, opposite/adjacent)
    elif x >= 250 and y <= 250:
        opposite = 250.0 - y
        adjacent = x - 250.0
        if adjacent == 0:
            adjacent = .01
        degree = np.arctan(opposite/adjacent) * 57.2958 + 270
        # print(degree, "degrees 3 ", opposite, adjacent, opposite/adjacent)
    else:
        opposite = x - 250.0
        adjacent = y - 250.0
        if adjacent == 0:
            adjacent = .01
        degree = np.arctan(opposite/adjacent) * 57.2958 + 180
        # print(degree, "degrees 4", opposite, adjacent, opposite/adjacent)

    return degree

#send commands about the desired end angle and number of steps to get there
def calc_rot(rad_current, rad_desired):
    #this is how many radians I need to move in total
    desired_transform = rad_desired - rad_current

    #I want to move one degree per frame at a frame rate of 40
    oneeighty = 180/57.2958

    #This is to make sure the direction I am turning is the most efficient way to turn
    if desired_transform < 0:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = -1 * (1 / (57.2958)) #1 degree per frame
        else:
            rotation_rte = (1 / (57.2958)) #1 degree per frame
    else:
        if abs(desired_transform) <= oneeighty: #Decide whether to turn clockwise or counter clockwise
            rotation_rte = (1/ (57.2958))
        else:
            rotation_rte = -1 * (1 / (57.2958)) #1 degree per frame

    #Number of steps
    desired_transform = (abs(desired_transform))

    if desired_transform > (np.pi):
        desired_transform = 2*np.pi - desired_transform

    num_steps = desired_transform / rotation_rte

    return int(abs(num_steps)), rotation_rte

num_steps = 0
while 1:
    #This gets my mouse pressed states
    mouse_state = pygame.mouse.get_pressed()
    display.fill(white)

    if mouse_state[0] == 1:
        # print("mouse pressed")
        sprites.append(pygame.mouse.get_pos())

        desired_degree = print_angle(sprites[-1][0], sprites[-1][1])
        radians_needed = (desired_degree / 57.2958)
        current_radians = (current_degree/ 57.2958)
        # print(current_degree, desired)
        num_steps, rotate_rte = calc_rot(current_radians, radians_needed)

    if num_steps > 0:
        ua_image, ua_rect = upperarm.rotate((rotate_rte))
        num_steps += -1

    else:
        ua_image, ua_rect = upperarm.rotate((0))



    ua_rect.center += np.asarray(base)

    ua_rect.center -= np.array([np.cos(upperarm.rotation) * upperarm.offset,
                                -np.sin(upperarm.rotation) * upperarm.offset])

    current_degree = print_angle(ua_rect.center[0], ua_rect.center[1])

    display.blit(ua_image, ua_rect)
    pygame.draw.circle(display, black, base, 30)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            print(sprites)
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(30)
