import glob
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import time

tf.compat.v1.disable_eager_execution()

i_sizes = [[1, 16, 16, 3, 5]]
red_type = ['MAX', 'SUM', 'MEAN', 'STD', 'VAR', 'MIN', 'PROD'] #, 'EUNORM', 'LOGSUMEXP'

desstr = '\nstd::vector<std::string> reduce_descriptions = {'
istr = '\nstd::vector<std::vector<float>> reduce_is = {'
ostr = '\nstd::vector<std::vector<float>> reduce_os = {'
modstr = '\nstd::vector<std::string> reduce_modules = {'

def ary2str(A):
    A = A.flatten()
    ret = '{'
    for i in range(len(A)):
        ret += str(A[i]) + ', '
    return ret[:-1]+ '}'

#Calculate convoluion for each combination; store inputs, outputs & module
for (i, combination) in enumerate(itertools.product(i_sizes, red_type)):
    I = tf.compat.v1.placeholder(tf.float32, combination[0])
    
    if combination[1] == 'EUNORM':
        R = tf.math.reduce_euclidean_norm(I, len(combination[0])-1)
    elif combination[1] == 'LOGSUMEXP':
        R = tf.math.reduce_logsumexp(I, len(combination[0])-1)
    elif combination[1] == 'MAX':
        R = tf.math.reduce_max(I, len(combination[0])-1)
    elif combination[1] == 'MEAN':
        R = tf.math.reduce_mean(I, len(combination[0])-1)
    elif combination[1] == 'MIN':
        R = tf.math.reduce_min(I, len(combination[0])-1)
    elif combination[1] == 'PROD':
        R = tf.math.reduce_prod(I, len(combination[0])-1)
    elif combination[1] == 'STD':
        R = tf.math.reduce_std(I, len(combination[0])-1)
    elif combination[1] == 'SUM':
        R = tf.math.reduce_sum(I, len(combination[0])-1)
    elif combination[1] == 'VAR':
        R = tf.math.reduce_variance(I, len(combination[0])-1)

    with tf.compat.v1.Session() as sess:
        ia = np.random.uniform(size = combination[0])
        result = sess.run(R, feed_dict={
            I: ia
        })

    desstr += '\n\"'
    for ci in range(len(combination)):
        desstr += str(combination[ci]).replace(', ','x')+'__'
    desstr += "\","
    istr += '\n'+ ary2str(ia) + ','
    ostr += '\n'+ ary2str(result) + ','
    modfile = open(glob.glob('tensorflow/compiler/xla/service/plaidml/tests/reduce_hlo_module/*.before_optimizations.txt')[0])
    module = modfile.read()
    modfile.close()
    modstr += '\nR\"#('+ module + ')#\",'

    #Clean module directory
    remdir = 'tensorflow/compiler/xla/service/plaidml/tests/reduce_hlo_module/'
    shutil.rmtree(remdir)

#Format & save header file
istr = istr[:-1] + '};'
ostr = ostr[:-1] + '};'
modstr = modstr[:-1] + '};'
desstr = desstr[:-1].replace('[','').replace(']','') +'};'

fstr ='\n' + desstr + istr + ostr + modstr

iofile = open('tensorflow/compiler/xla/service/plaidml/tests/plaidml_reduce_op_test.h.inc', 'w+')
iofile.write(fstr)
iofile.close()