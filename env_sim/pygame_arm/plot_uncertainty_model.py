import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats



def make_uncertainty_plots(h, h_2, p, p2):
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
    fit_2 = stats.norm.pdf(h_2, np.mean(h_2), np.std(h_2))

    x = plt.figure(1)
    # x.patch.set_facecolor('white')
    plt.subplot(211)
    plt.title("Theta 1")
    plt.plot(h,fit,'-o')
    plt.hist(h,normed=True)
    plt.axvline(x=p, lw=4)
    plt.yticks([])
    plt.xlim((0,6.5))
    plt.xlabel("Radians")
    plt.subplot(212)
    plt.title("Theta 2")
    plt.plot(h_2,fit_2,'-o')
    plt.hist(h_2,normed=True)
    plt.axvline(x=p2, lw=4)
    plt.yticks([])
    plt.xlim((0,6.5))
    plt.xlabel("Radians")
    plt.tight_layout()
    plt.show()



x = np.load("theta0.npy")
y = np.load("theta1.npy")

make_uncertainty_plots(x[:-1],y[:-1],x[-1],y[-1])
