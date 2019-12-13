
import matplotlib.pyplot as plt
import time
import numpy as np

x = range(1,10)
y = range(1,10)

plt.ion()
plt.show()

for i, j in zip(x,y):
    print(i,j)
    x = np.random.rand(10)
    y = np.random.rand(10)
    plt.plot(x, y, "--")

    x = np.random.rand(10)
    y = np.random.rand(10)
    plt.plot(x, y)
    plt.pause(0.01)
    plt.clf()
    time.sleep(3)
