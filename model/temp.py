import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 100)
y0 = 0
y1 = 1
z0 = -(y0 * np.log(x) + (1 - y0) * np.log(1 - x))
z1 = -(y1 * np.log(x) + (1 - y1) * np.log(1 - x))
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x, z0)
plt.subplot(1, 2, 2)
plt.plot(x, z1)
plt.show()