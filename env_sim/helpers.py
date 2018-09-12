import numpy as np
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import pygame.locals
import matplotlib
import matplotlib.backends.backend_agg as agg
import scipy.stats as stats
import matplotlib.pyplot as plt

'''Helper Functions'''
def convert_normal_angle(t_0, t_1):
    if t_0 < 0:
        t_0 = 2* np.pi + t_0
    if t_1 < 0:
        t_1 = 2* np.pi + t_1
    return t_0, t_1

def calc_rot(rad_current, rad_desired):
    #This is how many radians I need to move in total
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

def inv_kin_2arm(x, y, l0, l1):
    inside = (x**2 + y**2 - l0**2 - l1**2)/(2*l0*l1)
    inside = round(inside, 5)
    if (x**2 + y**2 )**.5 > l0 + l1 or abs(inside) > 1 or (x == 0 and y == 0):
        return -20, -20
    else:
        theta_1 = (np.arccos(inside))
        a = y * (l1 * np.cos(theta_1) + l0) - x * l1 * np.sin(theta_1)
        if x == 0:
            x = .00001
        b = x * (l1 * np.cos(theta_1) + l0) + y * l1 * np.sin(theta_1)

        if b == 0:
            return -20, -20
        theta_0 = np.arctan2(a, b)
    # print(theta_0, theta_1)
    return theta_0, theta_1

def return_ordered(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def make_uncertainty_plots(h, h_2, p, p2):
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
    fit_2 = stats.norm.pdf(h_2, np.mean(h_2), np.std(h_2))

    x = plt.figure(1)
    x.patch.set_facecolor('white')
    plt.subplot(211)
    plt.title("Theta 1")
    plt.plot(h,fit,'-o')
    plt.hist(h,normed=True)
    plt.yticks([])
    plt.xlim((0,6.5))
    plt.xlabel("Radians")
    plt.subplot(212)
    plt.title("Theta 2")
    plt.plot(h_2,fit_2,'-o')
    plt.hist(h_2,normed=True)
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
