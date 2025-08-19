#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

def softmax_activation(x):
    exp_x = np.exp(x - np.max(x))  # for numerical stability
    return exp_x / np.sum(exp_x)

# Perceptron Output Function
def perceptron_output(x, w, b, activation_func, is_vector=False):
    z = w * x + b
    if is_vector:
        return activation_func(z)
    else:
        return activation_func(z)

x_vals = np.linspace(-10, 10, 200)

weight = 1.0
bias = 0.0

linear_outputs = perceptron_output(x_vals, weight, bias, linear_activation)
sigmoid_outputs = perceptron_output(x_vals, weight, bias, sigmoid_activation)
tanh_outputs = perceptron_output(x_vals, weight, bias, tanh_activation)
relu_outputs = perceptron_output(x_vals, weight, bias, relu_activation)
softmax_outputs = perceptron_output(x_vals, weight, bias, softmax_activation, is_vector=True)

plt.figure(figsize=(12, 8))

plt.plot(x_vals, linear_outputs, label="Linear Activation", color='blue')
plt.plot(x_vals, sigmoid_outputs, label="Sigmoid Activation", color='green')
plt.plot(x_vals, tanh_outputs, label="Tanh Activation", color='red')
plt.plot(x_vals, relu_outputs, label="ReLU Activation", color='purple')
plt.plot(x_vals, softmax_outputs, label="Softmax Activation", color='orange')

plt.title("Perceptron Output with Different Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




