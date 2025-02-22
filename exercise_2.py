
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
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

A = 2 # amplitude of oscillations
g = 0.2 # damping coefficient
o = math.pi # period of oscillations
p = math.pi/2 * 0 # shift of period along x axis
stop = o*4
step = o/250
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
eps = np.linspace(0.1, 0.4, 4) # max distance from a point to another point to consider them one cluster
min_samples = [3, 5, 10, 30] # minimum number of points required to consider them a cluster
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
# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=528)

# initialization and training of the model
model1 = LinearRegression()
model1.fit(X_train, y_train)

model_sgd = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.01)
losses = []
r2_scores = [] 
# simulation of iterations for linear regression by method of stochastic gradient descent:
for i in range(100):
    model_sgd.partial_fit(X_train, y_train)  # incremental update of the model
    y_pred = model_sgd.predict(X_train) 
    loss_sgd = mean_squared_error(y_train, y_pred)  
    losses.append(loss_sgd)
    r2_scores.append(model_sgd.score(X_train, y_train))

# prediction based on test data
y_pred = model1.predict(X_test)

# evaluation of the model
print("intercept:", model1.intercept_)
print("coefficient:", model1.coef_[0])
print("mean squared error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.plot(range(1, 101), losses, label="MSE loss", color='blue')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Tracking linear regression loss over iterations (SGD)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 101), r2_scores, label="R² Score", color='red')
plt.xlabel("Iteration")
plt.ylabel("R² Score")
plt.title("R² Score Improvement over Iterations")
plt.legend()

plt.scatter(X, y, color = "blue", label = "actual data")
plt.plot(X, model1.predict(X), color = "red", label = "regression line")
plt.xlabel("time, s")
plt.ylabel("f(t)")
plt.legend()
plt.show()

print('Task completed; linear regression')

### NEURAL NETWORK ###
import torch
import torch.nn as nn
import torch.optim as optim

# making data 2D for correct processing by PyTorch:
x_train = torch.tensor(X.values, dtype = torch.float32).reshape(-1,1)
Y_train = torch.tensor(y.values, dtype = torch.float32).reshape(-1,1)

class NeuralNetwork(nn.Module):
    def __init__(self, input2 = 64, output1 = 64, input3 = 64, output2 = 64): 
        super().__init__() # calls the constructor of parent class (nn.Module) to ensure it's properly initialized
        self.fc1 = nn.Linear(1, output1) # layer 1: time as input -> conversion to many outputs
        self.fc2 = nn.Linear(input2, output2) # layer 2 (hidden/neuronal = not directly exposed to input or output): many inputs -> processing by neurons -> many outputs
        self.fc3 = nn.Linear(input3, 1) # layer 3: many inputs -> 1 output ( predicted oscillation amplitude at t)
        self.relu = nn.ReLU() # introduces non-linearity by replacing negative t values with 0
        # self.tanh = nn.Tanh() # alternative to ReLU 
    
    def forward(self, x):
        x = self.relu(self.fc1(x)) # applying first layer + ReLU
        x = self.relu(self.fc2(x)) # applying second layer + ReLU
        x = self.fc3(x) # applying final layer (output of regression)
        return x
   
model = NeuralNetwork()
criterion = nn.MSELoss() # loss function = Mean squared error
optimizer = optim.Adam(model.parameters(), lr = 0.001) # optimizer = Adam (adaptive moment estimation)

epochs = 1000 # limit for number of training epochs
epochs_store = [10, 100, 500, 999] # selected numbers of epochs to visualize in plot of output and error functions
predictions = {} # dictionary for storage of NN outputs for selected numbers of epochs passed 
losses_nn = [] # progress parameter storage for tracking

for epoch in range(epochs): # training for given maximum number of epochs
    optimizer.zero_grad() # clearing previous gradients
    output = model(x_train) # forward pass (prediction)
    loss_t = criterion(output, Y_train) # computation of loss (difference between prediction and actual Y)
    loss_t.backward() # computation of gradients of the loss function using backpropagation, given the model parameters 
    optimizer.step() # update of model parameters (weights and biases) based on computed gradients
    losses_nn.append(loss_t.item())  # storing loss value
    if epoch in epochs_store:
        predictions[epoch] = output.detach().numpy()
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss_t.item()}')

# Y_pred_nn = model(x_train).detach().numpy() # conversion of predictions to numpy
plt.figure(figsize=(20, 12))
plt.scatter(x_train, Y_train, label="Noisy data", alpha = 0.6)
# plt.plot(X_train, Y_pred_nn, label="Neural network output", color='r') # final prediction
for epoch, prediction in predictions.items():
    plt.plot(x_train, prediction, linewidth = 2.5, label = f'NN prediction at epoch {epoch}')
    plt.plot(x_train, prediction - Y_train.detach().numpy(), ':', label = f'Error at epoch {epoch}')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('NN regression at different epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(range(epochs), losses_nn, color='blue', label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Neural network training loss")
plt.legend()
plt.show()

print("Task completed; NN regression")

### PHYSICS-INFORMED NEURAL NETWORK (PINN) ###

class PINN(nn.Module):
    def __init__(self): 
        super().__init__() # calls the constructor of parent class (nn.Module) to ensure it's properly initialized
        self.fc1 = nn.Linear(1, 64) # layer 1: time as input -> conversion to 64 outputs
        self.fc2 = nn.Linear(64, 64) # layer 2 (hidden/neuronal = not directly exposed to input or output): 64 inputs -> processing by 64 neurons -> 64 outputs
        self.fc3 = nn.Linear(64, 64) # layer 3 (hidden/neuronal = not directly exposed to input or output): 64 inputs -> processing by 64 neurons -> 64 outputs
        self.fc4 = nn.Linear(64, 1) # layer 4: 64 inputs -> 1 output ( predicted oscillation amplitude at t)
        self.relu = nn.ReLU() # introduces non-linearity by replacing negative t values with 0
    
    def forward(self, t):
        x = self.relu(self.fc1(t)) 
        x = self.relu(self.fc2(x)) 
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def p_loss(model = PINN()):
    A = 2 # amplitude
    g = 0.2 # damping coefficient
    o = np.pi # period
    p = np.pi/2*0 # shift
    t = x_train.requires_grad_() # conversion of t to a gradient-tracking enabled tensor
    y_pinn = model(t) # forward pass

    # dy/dt (first derivative of the output with respect to time):
    dy_dt = torch.autograd.grad(y_pinn, t, grad_outputs=torch.ones_like(y_pinn), retain_graph=True)[0]

    f_phys = A * torch.exp(-g*t) + torch.sin (o*t + p) # physical function
    residual = dy_dt + g * y_pinn - f_phys  # residual for loss calculation
    return torch.mean((residual)**2) # physics-based loss term
    
def d_loss(model = PINN()):
    y_pred = model(x_train)
    return torch.mean((y_pred - Y_train)**2) # data-based loss term

def loss(model = PINN(), lambda_phy=10.0):
    D_loss = d_loss(model)
    P_loss = p_loss(model)
    # total loss (data loss + physics loss):
    return D_loss + lambda_phy * P_loss # lambda phy is the weight coefficient for physics-based loss
    # increase lampbda_phy if the network is not fitting well

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_list = []
d_loss_list = []
p_loss_list = []

epochs_pinn = 1000
epochs_pinn_store = [10, 100, 500, 999]
predictions_pinn = {} 
lambda_phy = 1.0 
for epoch in range(epochs_pinn):
    optimizer.zero_grad()
    TL = loss(model, lambda_phy)
    TL.backward()
    optimizer.step()

    pL = p_loss(model)
    pL.backward()
    optimizer.step()

    dL = d_loss(model)
    dL.backward()
    optimizer.step()

    loss_list.append(TL.item())
    d_loss_list.append(dL.item())
    p_loss_list.append(pL.item())

    if epoch in epochs_store:
        predictions_pinn[epoch] = TL.detach().numpy()
        print(f'Epoch [{epoch}/{epochs}], Loss: {TL.item()}')
    
with torch.no_grad():
    y_pred_pinn = model(x_train).detach().numpy()

plt.figure(figsize=(20, 12))
plt.scatter(x_train.detach().numpy(), Y_train.detach().numpy(), color="blue", alpha = 0.6, label="Noisy data")
for epoch, prediction in predictions.items():
    plt.plot(x_train.detach().numpy(), y_pred_pinn, linewidth = 2.5, label = f'PINN prediction at epoch {epoch}')
    plt.plot(x_train.detach().numpy(), y_pred_pinn - Y_train.detach().numpy(), ':', label = f'Error at epoch {epoch}')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('PINN regression at different epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), loss_list, label="Total loss", color='black')
plt.plot(range(epochs), d_loss_list, label="Data loss", color='blue')
plt.plot(range(epochs), p_loss_list, label="Physics loss", color='red')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN training losses")
plt.legend()
plt.show()

print("Task completed; PINN regression")



