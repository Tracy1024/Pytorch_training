
from pomegranate import BayesianNetwork
import matplotlib.pyplot as plt
import imageio
import numpy
import networkx
from pomegranate.utils import plot_networkx

X = imageio.imread("KW3NGK.jpg")
X = X[:, :, 1]
print(X.shape)
X = X[:, :31]
X = X > numpy.mean(X)

f1 = ()
f2 = ()
for i in range(0, 20):
    f1 += (i,)
for i in range(20, 30):
    f2 += (i,)

structure = ()
for i in range(0, 20):
    structure += ((), )
for i in range(20, 30):
    structure += (f1, )
for i in range(30, 31):
    structure += (f2, )
print(structure)

model = BayesianNetwork.from_structure(X, structure)
model.plot()
plt.show()
p = model.probability(X[:, 5])
print(p.shape)
print(p)
#plt.imshow(X)
#plt.show()