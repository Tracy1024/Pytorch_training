import torch as T
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = T.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = T.relu(x).data.numpy()
y_sigmoid = T.sigmoid(x).data.numpy()
y_tanh = T.tanh(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(311)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(312)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(313)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
plt.show()



