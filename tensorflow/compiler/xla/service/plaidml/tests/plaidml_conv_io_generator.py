import tensorflow as tf
import numpy as np
import os
import glob
import itertools
import shutil

np.set_printoptions(threshold = 125**2)
tf.compat.v1.disable_eager_execution()

i_sizes = [[1, 224, 224, 3]]
k_sizes = [[7, 7, 3, 64]]
strides = [(1, 1)]
paddings = [[[0, 0], [3, 3], [3, 3], [0, 0]]]

# [[0, 0], [0, 0], [0, 0], [0, 0]],
#             [[0, 0], [1, 1], [1, 1], [0, 0]],
#             "VALID",
#             "SAME",
            
dilations = [(1, 1)]

np.set_printoptions(threshold = max(np.product(np.array(i_sizes)), np.product(np.array(k_sizes))))

istr = '\nstd::vector<std::vector<float>> conv_is = {'
k1str = '\nstd::vector<std::vector<float>> conv_k1s = {'
k2str = '\nstd::vector<std::vector<float>> conv_k2s = {'
ostr = '\nstd::vector<std::vector<float>> conv_os = {'
modstr = '\nstd::vector<std::string> conv_modules = {'

nstr = '0000'

#Calculate convoluion for each combination; store inputs, outputs & module
for (i, combination) in enumerate(itertools.product(i_sizes, k_sizes, strides, paddings, dilations)):
    nstr = nstr[:3-(i//10)] + str(i)
    I = tf.compat.v1.placeholder(tf.float32, combination[0])
    K1 = tf.compat.v1.placeholder(tf.float32, combination[1])
    C1 = tf.nn.relu(tf.nn.conv2d(I, K1, strides = combination[2], padding = combination[3], dilations = combination[4]))
    k2sz = combination[1].copy()
    k2sz[2:] = combination[1][:1:-1]
    K2 = tf.compat.v1.placeholder(tf.float32, k2sz)
    C2 = tf.nn.relu(tf.nn.conv2d(C1, K2, strides = combination[2], padding = combination[3], dilations = combination[4]))

    with tf.compat.v1.Session() as sess:
        ia = np.random.uniform(size = combination[0])
        k1 = np.random.uniform(size = combination[1])
        k2 = np.random.uniform(size = k2sz)
        result = sess.run(C2, feed_dict={
            I: ia, 
            K1 : k1,
            K2 : k2
        })
    
    istr += '\n'+np.array2string(ia.flatten(), separator=',').replace('\n','') + ','
    k1str += '\n'+np.array2string(k1.flatten(), separator=',').replace('\n','') + ','
    k2str += '\n'+np.array2string(k2.flatten(), separator=',').replace('\n','') + ','
    ostr += '\n'+np.array2string(result.flatten(), separator=',').replace('\n','') + ','
    modfile = open(glob.glob('tensorflow/compiler/xla/service/plaidml/tests/conv_hlo_module/*'+nstr+'.before*')[0])
    module = modfile.read()
    modfile.close()
    modstr += '\nR\"#('+ module + ')#\",'

#Format & save header file
istr = istr.replace('[','{').replace(']','}')[:-1] + '};'
k1str = k1str.replace('[','{').replace(']','}')[:-1] + '};'
k2str = k2str.replace('[','{').replace(']','}')[:-1] + '};'
ostr = ostr.replace('[','{').replace(']','}')[:-1] + '};'
modstr = modstr[:-1] + '};'

fstr ='\n' + istr + k1str + k2str + ostr + modstr

iofile = open('tensorflow/compiler/xla/service/plaidml/tests/plaidml_conv_test_io.h', 'w+')
iofile.write(fstr)
iofile.close()

#Remove Unnecessary Files
remdir = 'tensorflow/compiler/xla/service/plaidml/tests/conv_hlo_module/'
try:
    shutil.rmtree(remdir)
except OSError as e:
    print(f'Error: {remdir} : {e.strerror}')