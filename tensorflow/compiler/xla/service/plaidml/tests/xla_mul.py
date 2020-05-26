import tensorflow as tf
tf.compat.v1.disable_eager_execution()

@tf.function(experimental_compile = True)
def mul(a, b):
  return a * b

with tf.compat.v1.Session() as sess:
    x = tf.constant([1.5, 0.5, -0.5, -1.5])
    with tf.device("/device:XLA_PLAIDML:0"):
        y = mul(x, x)
    result = sess.run(y)
    print(result)
