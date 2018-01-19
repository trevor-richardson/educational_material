import numpy as np
import matplotlib
import matplotlib.pyplot
import matplotlib.backends.backend_agg as agg
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

import pygame
from pygame.locals import *
h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
     187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
     161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted
#populate my

h_2 = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
     187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 220, 145, 210,
     430, 65, 72, 180, 99, 125, 600, 230, 152, 33, 25, 320, 420, 150])
# h = []
# h_2 = []
fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

fit_2 = stats.norm.pdf(h_2, np.mean(h_2), np.std(h_2))

x = matplotlib.pyplot.figure(1)
matplotlib.pyplot.subplot(211)
matplotlib.pyplot.title("Theta 1")
matplotlib.pyplot.plot(h,fit,'-o')
matplotlib.pyplot.hist(h,normed=True)
matplotlib.pyplot.xlabel("Radians")


matplotlib.pyplot.subplot(212)
matplotlib.pyplot.title("Theta 2")
matplotlib.pyplot.plot(h_2,fit_2,'-o')
matplotlib.pyplot.hist(h_2,normed=True)
matplotlib.pyplot.xlabel("Radians")
matplotlib.pyplot.tight_layout()

ax = matplotlib.pyplot.gca()

canvas = agg.FigureCanvasAgg(x)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()

pygame.init()

window = pygame.display.set_mode((600, 500), DOUBLEBUF)
screen = pygame.display.get_surface()

size = canvas.get_width_height()


surf = pygame.image.fromstring(raw_data, size, "RGB")
screen.blit(surf, (200,200))
pygame.display.flip()

crashed = False
while not crashed:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			crashed = True
