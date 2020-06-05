import tensorflow as tf
from tensorflow import keras

tf.compat.v1.disable_eager_execution()

@tf.function(experimental_compile=True)
def train_mnist(images, labels):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(images, labels, steps_per_epoch=10, epochs=10)
    return model

with tf.compat.v1.Session() as sess, tf.device("/device:XLA_PLAIDML:0"):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    model = train_mnist(train_images, train_labels)
