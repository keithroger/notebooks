# %%
'''
This notebook goes through the process of building a simple neural network.
'''

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# input data
fig, ax = plt.subplots()
x_1 = np.linspace(0, 3, 100)
y_1 = 1.5 * x_1 + np.random.normal(0, 1, 100)
x_2 = np.linspace(0, 3, 100)
y_2 = 1.5 * x_1 + np.random.normal(0, 1, 100) + 8
ax.scatter(x_1, y_1)
ax.scatter(x_2, y_2)
plt.title('Inpute Data')
plt.savefig('images/input_data.png')


# %%
'''
## Activation Functions
'''


# %%
def sigmoid(x):
    '''Sigmoid activation function'''
    return 1 / (1 + np.exp(-x))


# %%
def leakyReLU(x, alpha=0.3):
    '''LeakyRelu activation function'''
    return np.maximum(alpha * x, x)


# %%
def elu(x, alpha=1.0):
    '''ELU activation function'''
    # syntax used is to make the function compatible with numpy arrays
    return (x > 0) * x + (x <= 0) * (alpha * (np.exp(x) - 1))


# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
x = np.linspace(-7, 7, 100)
ax1.plot(x, sigmoid(x))
ax1.set_title('Sigmoid')
ax2.plot(x, leakyReLU(x))
ax2.set_title('LeakyReLU')
fig.suptitle('Activation Functions')
ax3.plot(x, np.tanh(x))
ax3.set_title('tanh(x)')
ax4.plot(x, elu(x))
ax4.set_title('ELU')
plt.tight_layout()
plt.savefig('images/activation_fuctions.png')


# TODO: create a function to show decision boundries
