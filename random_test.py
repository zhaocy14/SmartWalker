import numpy as np
import tensorflow as tf

a = tf.constant([1,2,3,4])
b = 10
c = tf.math.multiply(a,b)
print(c)

# d = tf.math.scalar_mul(a,b)
# print(c)