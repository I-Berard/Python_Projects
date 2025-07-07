import numpy as np
import matplotlib.pyplot as plt
import time 

tic = time.time()

X_raw = np.array([1, 2, 3, 4, 5])
y = np.array((1, 3, 3, 2, 5))

m = len(y)
X = np.c_[np.ones(m), X_raw]  # Concatenate the two columns, the one of our raw data and the one of the ones
y = y.reshape(-1, 1)    

beta = np.linalg.inv(X.T @ X) @ X.T @ y 

input_raw = np.array([3, 4, 1])
input = np.c_[np.ones(len(input_raw)), input_raw]

pred = input @ beta

print("Hours Studied | Predicted Score")
print("-------------------------------")
for n, m in zip(input_raw, pred.ravel()):
    print(f"{n:13} | {m:.2f}")

toc = time.time()

print(f"Time elapsed {toc - tic:.4f} seconds")