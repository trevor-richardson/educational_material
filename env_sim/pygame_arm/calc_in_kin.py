import numpy as np
import math


def inv_kin_2arm(x, y, l0, l1):
    print(x, y, "xy")
    inside = (x**2 + y**2 - l0**2 - l1**2)/(2*l0*l1)
    inside = round(inside, 5)

    if (x**2 + y**2 )**.5 > l0 + l1 or abs(inside) > 1 or x == 0 or y == 0:
        print("impossible to reach", l0, l1, x, y, abs(((x**2 + y**2) - l0**2 - l1**2)/(2*l0*l1)))
        print ((x**2 + y**2 - l0**2 - l1**2)/(2*l0*l1))
        return -1, -1
    else:
        theta_1 = (math.acos(inside))

        a = y * (l1 * math.cos(theta_1) + l0) - x * l1 * math.sin(theta_1)
        b = x * (l1 * math.cos(theta_1) + l0) + y * l1 * math.sin(theta_1)

        if b == 0:
            print("two ")
            print("impossible to reach", l0, l1, x, y, abs(((x^2 + y^2) - l0^2 - l1^2)/(2*l0*l1)))
            return -1, -1


        theta_0 = np.arctan2(a, b)

    return theta_0, theta_1

# theta_0 45, theta1 = 0 Correct
# x = 2.0 * (2.0 ** (.5))
# y = 2.0 * (2.0 ** (.5))

# theta0 -45, theta1 = 0
# x = -2.0 * (2.0 ** (.5))
# y = 2.0 * (2.0 ** (.5))

# theta_0 45, theta1 = 0
# x = -2.0 * (2.0 ** (.5))
# y = -2.0 * (2.0 ** (.5))

# theta_0 -45, theta1 = 0
# x = 2.0 * (2.0 ** (.5))
# y = -2.0 * (2.0 ** (.5))

'''quad 1-- '''
# x =  (2.0 ** (.5))
# y =  (2.0 ** (.5)) + 2


'''quad 2-- '''
# x =  -(2.0 ** (.5))
# y =  (2.0 ** (.5)) + 2


'''quad 3-- '''
# x =  -(2.0 ** (.5))
# y =  -((2.0 ** (.5)) + 2)


'''quad 4-- '''
# x =  (2.0 ** (.5))
# y =  -((2.0 ** (.5)) + 2)


th0, th1 = inv_kin_2arm(x, y, 2, 2)
print(th0 * 57.2958, th1 * 57.2958)
