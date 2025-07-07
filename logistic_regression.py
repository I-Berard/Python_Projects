import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0.5, 1.5],
    [1.0, 1.8],
    [1.5, 0.5],
    [3.0, 2.5]
])

y = np.array([0, 0, 1, 1])  

w = np.zeros(X.shape[1])
b = 0.0
error_log = []
learning_rate = 0.01

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calc_loss(y_pred, y_true):
    m = len(y_true)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return (-1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

m = len(y)

iterations = 2000

for i in range(iterations):
    pred = predict(X, w, b)

    error = calc_loss(pred, y)

    db = (1/m) * np.sum(pred - y)
    dw = (1/m) * np.dot(X.T, (pred - y))

    b -= learning_rate * db
    w -= learning_rate * dw

    error_log.append(error)
    # if(i % 10 == 0):
        # print(error_log[i])



print("Final weights:", w)
print("Final bias:", b)

# plt.figure(figsize=(8, 5))
# plt.plot(range(iterations), error_log, color='red')
# plt.xlabel("Iterations")
# plt.ylabel("Error")
# plt.title("Error over iterations")
# plt.grid(True)
# plt.show()

input = np.array([
    [3.0, 2.5],
    [1.0, 0.5]
])

output = predict(input, w, b)
output = output.reshape(-1,1)
print(output)

for i in range(len(output)):
    if output[i][0] < 0.5:
        print(f"Student {i + 1} failed the exam")
    elif output[i][0] >= 0.5:
        print(f"Student {i + 1} passed the exam")
    else:
        print("There has been a problem")


