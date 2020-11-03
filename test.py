import numpy as np

f1 = np.loadtxt("feature1.txt")
f2 = np.loadtxt("feature2.txt")

f1 = f1.reshape(f1.shape[0], f1.shape[1] // 20, 20)
f2 = f2.reshape(f2.shape[0], f2.shape[1] // 10, 10)

f1_fla = f1.reshape(f1.shape[0]*f1.shape[1], f1.shape[2])
f2_fla = f2.reshape(f2.shape[0]*f2.shape[1], f2.shape[2])
print(f1_fla.shape)
print(f2_fla.shape)

X = np.zeros((f1_fla.shape[0], f1_fla.shape[1]+f2_fla.shape[1]+1))
X[:, :f1_fla.shape[1]] = f1_fla
X[:f2_fla.shape[0], f1_fla.shape[1]:-1] = f2_fla
print(X.shape)