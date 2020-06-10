import tensorflow as tf
from tensorflow.keras.applications import ResNeXt50
import numpy as np

tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:
    model = ResNeXt50()
