
from pomegranate import BayesianNetwork
import matplotlib.pyplot as plt
import imageio
import numpy
import networkx
from pomegranate.utils import plot_networkx

num_f1 = 20
num_f2 = 25
X = imageio.imread("KW3NGK.jpg")
X = X[:, :, 1]
print(X.shape)
X = X[:, :num_f2]
X = X > numpy.mean(X)
plt.imshow(X)
plt.show()
f1 = ()
f2 = ()
for i in range(0, num_f1):
    f1 += (i,)
for i in range(num_f1, num_f2):
    f2 += (i,)

structure = ()
for i in range(0, num_f1):
    structure += ((), )
for i in range(num_f1, num_f2):
    structure += (f1, )
#structure += (f2, )
#for i in range(30, 31):
#    structure += (f2, )
print(structure)

model = BayesianNetwork.from_structure(X, structure)
model.plot()
plt.show()
p = model.probability(X)
print(p.shape)
print(p)
#plt.imshow(X)
#plt.show()