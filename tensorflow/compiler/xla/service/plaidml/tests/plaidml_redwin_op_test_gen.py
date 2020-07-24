import glob
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import time

opname = 'redwin'

tf.compat.v1.disable_eager_execution()

i_sizes = [[1, 16, 16, 3]]
pool_type = ['MAX'] #, 'AVG
k_sizes = [[1, 1], [2, 2], [1], [1, 3, 3, 1]]
strides = [[1, 1], [2, 2], [1], [1, 3, 3, 1]]
padding = ['VALID', 'SAME']

desstr = '\nstd::vector<std::string> '+opname+'_descriptions = {'
istr = '\nstd::vector<std::vector<float>> '+opname+'_is = {'
ostr = '\nstd::vector<std::vector<float>> '+opname+'_os = {'
modstr = '\nstd::vector<std::string> '+opname+'_modules = {'

def ary2str(A):
    A = A.flatten()
    ret = '{'
    for i in range(len(A)):
        ret += str(A[i]) + ', '
    return ret[:-1]+ '}'

#Calculate convoluion for each combination; store inputs, outputs & module
for (i, combination) in enumerate(itertools.product(i_sizes, pool_type, k_sizes, strides, padding)):
    I = tf.compat.v1.placeholder(tf.float32, combination[0])
    
    if combination[1] == 'MAX':
        R = tf.nn.max_pool(I, ksize = combination[2], strides = combination[3], padding = combination[4])
    elif combination[1] == 'AVG':
        R = tf.nn.avg_pool(I, ksize = combination[2], strides = combination[3], padding = combination[4])

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
    modfile = open(glob.glob('tensorflow/compiler/xla/service/plaidml/tests/'+opname+'_hlo_module/*.before*')[0])
    module = modfile.read()
    modfile.close()
    modstr += '\nR\"#('+ module + ')#\",'

    #Clean module directory
    remdir = 'tensorflow/compiler/xla/service/plaidml/tests/'+opname+'_hlo_module/'
    shutil.rmtree(remdir)

#Format & save header file
istr = istr[:-1] + '};'
ostr = ostr[:-1] + '};'
modstr = modstr[:-1] + '};'
desstr = desstr[:-1].replace('[','').replace(']','') +'};'

fstr ='\n' + desstr + istr + ostr + modstr

iofile = open('tensorflow/compiler/xla/service/plaidml/tests/plaidml_'+opname+'_op_test.h.inc', 'w+')
iofile.write(fstr)
iofile.close()