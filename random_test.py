import numpy as np
a = np.array([[1,2,3,4,5,6]])
pos = np.unravel_index(np.argmax(a),a.shape)
print(pos)
# d = tf.math.scalar_mul(a,b)
# print(c)