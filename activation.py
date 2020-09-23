from matplotlib import pyplot as plt
import numpy as np

z = np.arange(-6.1, 6.1, .1)
zero = np.zeros(len(z))
relu = np.max([zero, z], axis=0)

tanh = np.tanh(z)
sigmoid = 1 / (1+(np.e**(-z)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, relu, label="ReLU")
ax.plot(z, tanh, label="tanh")
ax.plot(z, sigmoid, label="sigmoid")
ax.legend(loc="upper left")
ax.set_ylim([-1.5, 1.5])
ax.set_xlim([-4, 4])
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('σ(x)')
ax.set_title('Activation Functions')

plt.savefig("a.png")
