import glob
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import time

tf.compat.v1.disable_eager_execution()

opsetname = 'shape'

ops = ['pad'] #'broadcast', 'reshape', 

def getInputs(opname):
    if opname == 'broadcast':
        # Input shape, output shape
        tests = [[[1, 3], [3, 3]],
                 [[5, 1, 2], [5, 3, 2]]]
        def opfunc(test):
            I = tf.compat.v1.placeholder(tf.float32, test[0])
            O = tf.broadcast_to(I, test[1])
            with tf.compat.v1.Session() as sess:
                i = np.random.uniform(size = test[0])
                o = sess.run(O, feed_dict={
                    I: i
                })
            return [i], [o]
        
        return tests, opfunc
    elif opname == 'reshape':
        # Product of each isize must be divisible by product of each odim
        i_sizes = [[12, 12], [12, 12, 12], [24]]
        o_dims = [[1], [2], [6], [3, 4], [2, 3, 1], [12], [3, -1, 2], [-1, 12]]
        tests = itertools.product(i_sizes, o_dims)
        def opfunc(test):
            if np.any(np.less(test[1], 0)):
                osize = test[1]
            else:
                fdim = np.product(test[0])// np.product(test[1])
                osize = test[1] + [fdim]
            if len(osize) == len(test[0]):
                osize = osize + [1] # Ensure the reshape is not a no-op
            I = tf.compat.v1.placeholder(tf.float32, test[0])
            O = tf.reshape(I, osize)
            with tf.compat.v1.Session() as sess:
                i = np.random.uniform(size = test[0])
                o = sess.run(O, feed_dict={
                    I: i
                })
            return [i], [o]
        
        return tests, opfunc
    elif opname == 'pad':
        # Product of each isize must be divisible by product of each odim
        i_sizes = [[2, 2]]
        max_pads = [3]
        modes = ['CONSTANT']
        cvals = [1]
        tests = itertools.product(i_sizes, max_pads, modes, cvals)
        def opfunc(test):
            padding = np.random.randint(test[1], size = (len(test[0]),2))
            if test[2] == 'REFLECT':
                padding = np.minimum(padding, np.expand_dims(np.array(test[0]),1) - 1)
            elif test[2] == 'SYMMETRIC':
                padding = np.minimum(padding, np.expand_dims(np.array(test[0]),1))
            I = tf.compat.v1.placeholder(tf.float32, test[0])
            O = tf.pad(I, padding, test[2], test[3])
            with tf.compat.v1.Session() as sess:
                i = np.random.uniform(size = test[0])
                o = sess.run(O, feed_dict={
                    I: i
                })
            return [i], [o]
        
        return tests, opfunc
    pass

def ary2str(A):
    A = A.flatten()
    ret = '{'
    for i in range(len(A)):
        ret += str(A[i]) + ', '
    return ret[:-2]+ '}'

fstr = ''

for opname in ops:
    tests, opfunc = getInputs(opname)

    desstr = 'std::vector<std::string> ' + opname + '_descriptions = {'
    istr = '\nstd::vector<std::vector<std::vector<float>>> ' + opname + '_is = {'
    ostr = '\nstd::vector<std::vector<std::vector<float>>> ' + opname + '_os = {'
    modstr = '\nstd::vector<std::string> ' + opname + '_modules = {'

    for test in tests:
        inputs, outputs = opfunc(test)
        
        desstr += '\n\"'
        for ti in test:
            desstr += str(ti).replace(', ','x')+'__'
        desstr = desstr[:-2] + "\","
        istr += '\n{'
        for inp in inputs:
            istr += ary2str(inp) + ','
        istr = istr[:-1] + '},'
        ostr += '\n{'
        for outp in outputs:
            ostr += ary2str(outp) + ','
        ostr = ostr[:-1] + '},'
        modfile = open(glob.glob('tensorflow/compiler/xla/service/plaidml/tests/hlo_module/*.before_optimizations.txt')[0])
        module = modfile.read()
        modfile.close()
        modstr += '\nR\"#('+ module + ')#\",'

        remdir = 'tensorflow/compiler/xla/service/plaidml/tests/hlo_module/'
        shutil.rmtree(remdir)

    #Format & save header file
    istr = istr[:-1] + '};\n'
    ostr = ostr[:-1] + '};\n'
    modstr = modstr[:-1] + '};\n'
    desstr = desstr[:-1].replace('[','').replace(']','') +'};\n'

    fstr += desstr + istr + ostr + modstr + '\n\n'

iofile = open('tensorflow/compiler/xla/service/plaidml/tests/plaidml_' + opsetname + '_op_test.h.inc', 'w+')
iofile.write(fstr)
iofile.close()