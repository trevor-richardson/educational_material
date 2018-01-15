import numpy as np
import pygame
import pygame.locals

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
white = (255, 255, 255)

pygame.init()

width = 500
height = 500
display = pygame.display.set_mode((width, height))
fpsClock = pygame.time.Clock()

upperarm = ArmPart('upperarm.png', scale=.7)

base = (int(width / 2), int(height / 2))

while 1:

    display.fill(white)

    ua_image, ua_rect = upperarm.rotate(.01)
    print(ua_rect, ua_image)
    ua_rect.center += np.asarray(base)
    ua_rect.center -= np.array([np.cos(upperarm.rotation) * upperarm.offset,
                                -np.sin(upperarm.rotation) * upperarm.offset])

    display.blit(ua_image, ua_rect)

    print(base)
    pygame.draw.circle(display, black, base, 30)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(100)
