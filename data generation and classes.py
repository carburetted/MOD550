import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(seed=1)
m = rng.standard_normal(15)
x = np.arange(15)
y = 0.5*x +2+m
print(m)

plt.figure(figsize=(8, 6))
plt.plot(x, y, '-r', label='Linear function with noise')
plt.scatter(x, y, c ='blue', label='Data points')
plt.title('Random Data with Noise')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

x_random = rng.uniform(0, 15, 100) 
y_random = rng.uniform(0, 10, 100)

plt.figure(figsize=(8, 6))
plt.scatter(x_random, y_random, c ='green', alpha=0.7, edgecolor='black')
plt.title('2D Random Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()