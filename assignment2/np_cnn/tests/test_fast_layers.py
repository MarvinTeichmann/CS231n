"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import scipy as scp

import time

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


from np_cnn import layers as nl
from np_cnn import fast_layers as fl


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def test_run_conv(verbose=True):
    x = np.random.randn(100, 3, 31, 31)
    w = np.random.randn(25, 3, 3, 3)
    b = np.random.randn(25,)
    # dout = np.random.randn(100, 25, 16, 16)
    conv_param = {'stride': 1, 'pad': 1}

    out_fast, cache_fast = fl.conv_forward_fast(x, w, b, conv_param)

    return


def test_fast_conv(verbose=True):
    x = np.random.randn(100, 3, 31, 31)
    w = np.random.randn(25, 3, 3, 3)
    b = np.random.randn(25,)
    dout = np.random.randn(100, 25, 16, 16)
    conv_param = {'stride': 2, 'pad': 1}

    t0 = time.time()
    out_naive, cache_naive = nl.conv_forward_naive(x, w, b, conv_param)
    t1 = time.time()
    out_fast, cache_fast = fl.conv_forward_fast(x, w, b, conv_param)
    t2 = time.time()

    if verbose:
        print('Testing conv_forward_fast:')
        print('Naive: %fs' % (t1 - t0))
        print('Fast: %fs' % (t2 - t1))
        print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
        print('Difference: ', rel_error(out_naive, out_fast))

    assert(rel_error(out_naive, out_fast) < 1e-9)

    t0 = time.time()
    dx_naive, dw_naive, db_naive = nl.conv_backward_naive(dout, cache_naive)
    t1 = time.time()
    dx_fast, dw_fast, db_fast = fl.conv_backward_fast(dout, cache_fast)
    t2 = time.time()


def test_fast_cython(verbose=True):
    x = np.random.randn(100, 3, 31, 31)
    w = np.random.randn(25, 3, 3, 3)
    b = np.random.randn(25,)
    conv_param = {'stride': 2, 'pad': 1}

    t0 = time.time()
    out_fast, cache_fast = fl.conv_forward_fast(x, w, b, conv_param)
    t1 = time.time()
    out_cython, cache_cython = fl.conv_forward_cython(x, w, b, conv_param)
    t2 = time.time()

    if verbose:
        print('Testing conv_forward_fast:')
        print('Cython: %fs' % (t1 - t0))
        print('Fast: %fs' % (t2 - t1))
        print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
        print('Difference: ', rel_error(out_cython, out_fast))

    assert(rel_error(out_cython, out_fast) < 1e-9)


if __name__ == '__main__':
    test_fast_cython(verbose=True)
    logging.info("Hello World.")
