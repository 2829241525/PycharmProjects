import numpy as np
import matplotlib.pyplot as plt


def adjusted_realistic_score_function(x, a=50, b=0.5, c=10):
    # Calculate initial values without noise
    initial_values = a * np.log(b * x + 1) + c
    # Normalize to the range with a maximum of 100 for x = 10 hours
    max_value_at_10h = a * np.log(b * 10 + 1) + c  # Maximum score at 10 hours based on the original function
    normalized_values = 100 * initial_values / max_value_at_10h  # Normalizing scores to max out at 100
    return initial_values

# Applying the adjusted function
X = np.array([0.01 * i for i in range(1001)])  # Extending range to include 10 hours
Y = adjusted_realistic_score_function(X)

# plt.scatter(X, Y, color="red", label="Original Data")  # 绘制原始数据点
# plt.legend()
# plt.show()


# Optimized training code

# Vectorized prediction function
def vectorized_init_func(X):
    return w1 * np.log(w2 * X + 1) + w3


# Training loop with optimizations
lr = 0.0001
w1, w2, w3 = 1,1,1
n_samples = len(X)

for epoch in range(10000):  # Reduced number of epochs for demonstration
    # Vectorized predictions and loss computation
    Y_pred = vectorized_init_func(X)
    errors = Y_pred - Y
    epoch_loss = np.mean(errors ** 2)

    # Vectorized gradient computation
    grad_w1 = 2 * np.mean(errors * np.log(w2 * X + 1))
    grad_w2 = 2 * np.mean(errors * w1 * X / (w2 * X + 1))
    grad_w3 = 2 * np.mean(errors)

    # Weight updates
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
    w3 -= lr * grad_w3

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss}")

    # Early stopping condition
    if epoch_loss < 0.001:
        print(f"Early stopping at epoch {epoch} with loss {epoch_loss}")
        break

print(f"Final weights: w1: {w1}, w2: {w2}, w3: {w3}")

# Plotting
plt.figure(figsize=(10, 6))
Y_pred = vectorized_init_func(X)  # 计算预测值
plt.scatter(X, Y, color="red", label="Original Data")  # 绘制原始数据点
plt.scatter(X, Y_pred, color="black",  label="Predicted Data")  # 增加线宽
#plt.scatter(X, Y_pred, color="black",  label="Predicted Data")  # 增加线宽
plt.legend()
plt.show()


