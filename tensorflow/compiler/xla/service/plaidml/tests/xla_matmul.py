import tensorflow as tf
tf.compat.v1.disable_eager_execution()

@tf.function(experimental_compile = True)
def mmul(a, b):
  return tf.matmul(a, b)

with tf.compat.v1.Session() as sess:
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with tf.device("/device:XLA_PLAIDML:0"):
        z = mmul(x, y)
    result = sess.run(z)
    print(result)
