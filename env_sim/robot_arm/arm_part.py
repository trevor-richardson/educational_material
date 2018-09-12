
'''Simple class for storing pictures and edicting the rect in PyGame'''
import numpy as np
import pygame
import pygame.locals

class ArmPart:
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
