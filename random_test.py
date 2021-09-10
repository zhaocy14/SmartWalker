import numpy as np

a = np.ones((3,2))
b = np.zeros((3,4))

c = np.concatenate([a,b],axis=1)
print(c)