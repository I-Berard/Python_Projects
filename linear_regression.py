import numpy as np
import matplotlib.pyplot as plt

X_raw = np.array([1, 2, 3, 4, 5])
y = np.array((1, 3, 3, 2, 5))

m = len(y)

X = np.c_[np.ones(m), X_raw] #concatenate the two columns, the one of our raw data and the one of the ones 
y = y.reshape(-1, 1)

alpha = 0.01 #learning rate
epochs = 1000
beta = np.zeros((2, 1))
cost_history = []


for i in range(epochs):
    pred = X @ beta
    error = pred - y
    gradient = (2/m) * X.T @ error
    beta -= alpha * gradient

    cost = (1/m) * np.sum(error ** 2)
    cost_history.append(cost)

plt.figure(figsize=(8, 5))
plt.scatter(X_raw, y, color='blue', label='Data')
plt.plot(X, X @ beta, color='red', label='Fitted line')
plt.xlabel("Hours studied")
plt.ylabel("Exam scored")
plt.title("Linear regression Fit")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(epochs), cost_history, color='green')
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.grid(True)
plt.show()


# print(cost_history)
