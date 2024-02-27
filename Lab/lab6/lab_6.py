import numpy as np
import matplotlib.pyplot as plt

h = 0.01*2.**(-np.arange(0, 10))

print(h)

forward = (np.cos(np.pi/2.+h)-np.cos(np.pi/2.))/h
centered = (np.cos(np.pi/2.+h)-np.cos(np.pi/2.-h))/(2.*h)

plt.plot(forward, label = "forward")
plt.plot(centered, label = "centered")
plt.legend()

print(forward)

"""
For forward difference the order of convergence is: O(h)
    The Truncation error is T(h) = f'(0) - (f(h)-f(0))/h
                                 = -h/2 f''(0) + O(h^2) 
                                 = O(h)
For Centered Difference the order of convergence is: O(h^2)
    The Truncation error is T(h) = f'(0) = (f(h)-f(-h))/2h
                                 = -[(h^3/3)f'''(0) + O(h^4)]/2h + O(h^2)
                                 = O(h^2)
"""