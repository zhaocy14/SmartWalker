def my_inter(x, x0, x1, y0, y1):
    return (x - x1) / (x0 - x1) * y0 + (x - x0) / (x1 - x0) * y1
print(my_inter(0.42,0.4,0.45,150,140))
# d = tf.math.scalar_mul(a,b)
# print(c)