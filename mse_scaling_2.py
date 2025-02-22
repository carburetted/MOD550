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

