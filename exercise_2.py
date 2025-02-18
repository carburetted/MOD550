
############  FIXING OF MSE_SCALING_2.PY CODE ############

from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
import timeit as it
import inspect

print(f'vanilla_mse expects argument names: {inspect.signature(vanilla_mse)}')
print(f'numpy_mse expects argument names: {inspect.signature(numpy_mse)}')
print(f'sk_mse expects argument names: {inspect.signature(sk_mse)}')

observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]
karg = {
    "mse_vanilla": {'observed': observed, 'predicted': predicted},
    "mse_numpy": {'observed': observed, 'predicted': predicted},
    "mse_sk": {'y_true': observed, 'y_pred': predicted}
    }
factory = {'mse_vanilla' : vanilla_mse,
    'mse_numpy' : numpy_mse,
    'mse_sk' : sk_mse
    }

for talker, worker in factory.items():
    exec_time = it.timeit(lambda: worker(**karg[talker]), number=100) / 100
    mse = worker(**karg[talker])
    print(f"Mean Squared Error, {talker} :", mse, 
          f"Average execution time: {exec_time:.8f} seconds")

if factory['mse_vanilla'](**karg["mse_vanilla"]) == factory['mse_numpy'](**karg["mse_numpy"]) == factory['mse_sk'](**karg["mse_sk"]):
    print('Test successful')

############  GENERATION OF DATA ############

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

A = 2 # amplitude of oscillations
g = 0.2 # damping coefficient
o = math.pi # period of oscillations
p = math.pi/2 * 0 # shift of period along x axis
stop = o*6
step = o/100
t = np.arange(0, stop, step) # time array

# generation of noise component:
rng = np.random.default_rng(seed=528)
m = rng.normal(0, 0.1, len(t))

'''m = np.random.normal(0, 0.1, len(t))  # works fine without rng instance, IF we do NOT need reproducibility'''

def f(t):
    return A * np.exp(-g * t) * np.sin(o * t + p) # damped oscillations function
Y = f(t) + m[:len(t)] # added noise

print(f'Data generated: {stop/step} points, ranged from 0 to {stop}. Oscillations amplitude: {A}, damping coefficient: {g}')

############  CLUSTERING OF THE DATA ############

data = {"time": t, "f(t)": Y}
df = pd.DataFrame(data) # dataframe 

# standardization of the data / feature scaling: 
# transforms data to match standard distribution - mean of 0 and standard deviation of 1
scaler = StandardScaler()
df_st = df.copy() # copy of the original dataset, to keep it as is
df_st[["time", "f(t)"]] = scaler.fit_transform(df[["time", "f(t)"]])

# vizualization before and after standardization of data:
plt.figure(figsize = (10,10))
plt.plot(t, f(t), label = "Damped oscillations without noise", c = 'k')
plt.scatter(t, Y, label = "Damped oscillations with noise", c = 'm')
plt.scatter(df_st["time"], df_st["f(t)"], label = "Standardized data", c = 'b', alpha = 0.5)
plt.legend()
plt.xlabel('time, s')
plt.ylabel('f(t)')
plt.grid(True)
plt.show()

print("Noisy dataset before standardization:", df.head())
print("Noisy dataset after standardization:", df_st.head())

# clustering with DBSCAN (density based spatial clustering of applications with noise):
# automates finding optimal number of clusters, where it is hard to set manually, like when forms of both function and clusters are complex.
eps = np.linspace(0.1, 0.5, 5) # max distance from a point to another point to consider them one cluster
min_samples = [3, 5, 10] # minimum number of points required to consider them a cluster
variances = [] # array of variances to be measured by different numbers of clusters
clusters = [] # array of different numbers of clusters
X = df_st[["time"]]
y = df_st["f(t)"]
for i in eps:
    for j in min_samples:
        db = DBSCAN(eps=i, min_samples=j).fit(df_st) # fitting the model
        labels = db.labels_ # extracting cluster labels (-1 means outliers)
        # labels = DBSCAN(eps=eps, min_samples=5).fit_predict(df_st) # same result, but with fitting and labelling in one step
        labels_true = set(labels) - {-1} # excluding outliers from labels

        clusters.append(len(labels_true)) # storing number of clusters
    
        variances.append(
           np.mean([np.var(X[labels == label]) for label in labels_true if sum(labels == label) > 1]) or 0
        )   # storing variances
        # plotting results of clustering for each combination of eps and min_samples:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, c=labels, cmap="viridis", edgecolors="k", label="DBSCAN clusters") # plotting clusters in different colors
        plt.scatter(X[labels == -1], y[labels == -1], color = 'red', label = 'Noise', edgecolor='k') # highlighting noise points
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.title(f"DBSCAN clustering of damped oscillations with noise. Maximum distance between points within a cluster: {i}, minimum points in a cluster: {j}")
        plt.colorbar(label="cluster #")
        plt.legend()
        plt.show()

# plotting variance as function of number of clusters:
plt.figure(figsize = (10, 6))
plt.plot(clusters, variances, marker='o')
plt.xlabel("number of clusters")
plt.ylabel("average variance within clusters")
plt.grid(True)
plt.show()

############  REGRESSION OF THE DATA ############

### LINEAR REGRESSION ###
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=528)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("intercept:", model.intercept_)
print("coefficient:", model.coef_[0])
print("mean squared error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(X, y, color = "blue", label = "actual data")
plt.plot(X, model.predict(X), color = "red", label = "regression line")
plt.xlabel("time, s")
plt.ylabel("f(t)")
plt.legend()
plt.show()

print('Task completed; linear regression')

### NEURAL NETWORK ###