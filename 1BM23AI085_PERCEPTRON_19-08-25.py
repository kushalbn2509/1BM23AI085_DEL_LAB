#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def linear_activation(x):
    return x

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return np.maximum(0, x)

def leaky_relu_activation(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def perceptron_output(x, w, b, activation_func):
    z = w * x + b
    return activation_func(z)

x_vals = np.linspace(-10, 10, 300)

weight = 1.0
bias = 0.0

outputs = {
    "Linear": perceptron_output(x_vals, weight, bias, linear_activation),
    "Sigmoid": perceptron_output(x_vals, weight, bias, sigmoid_activation),
    "Tanh": perceptron_output(x_vals, weight, bias, tanh_activation),
    "ReLU": perceptron_output(x_vals, weight, bias, relu_activation),
    "Leaky ReLU": perceptron_output(x_vals, weight, bias, leaky_relu_activation),
}

plt.figure(figsize=(12, 8))

for name, y_vals in outputs.items():
    plt.plot(x_vals, y_vals, label=f"{name} Activation")

plt.title("Perceptron Output with Linear and Non-Linear Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




