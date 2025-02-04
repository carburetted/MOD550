import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(seed=1)
m = = rng.standard_normal(8)
x = np.arange(15)
y = 0.5*x + 2 + m
print(m)

plt.figure()
plt.plot(x,y, '-r')
plt.show()
