import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:
    x_in = np.array([[
                    [[2], [1], [2], [0], [1]],
                    [[1], [3], [2], [2], [3]],
                    [[1], [1], [3], [3], [0]],
                    [[2], [2], [0], [1], [1]],
                    [[0], [0], [3], [1], [2]], ]])
    kernel_in = np.array([
                        [ [[2, 0.1]], [[3, 0.2]] ],
                        [ [[0, 0.3]],[[1, 0.4]] ], ])
    x = tf.constant(x_in, dtype=tf.float32)
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    out = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    print(out)
