# -*- coding: utf-8 -*-
""" CIS4930/6930 Applied ML --- utils.py
"""

import json
import re
import os
import time

import numpy as np


def train_test_val_split(x, y, prop_vec, shuffle=True, seed=None):

    assert x.shape[0] == y.shape[0]
    prop_vec = prop_vec / np.sum(prop_vec) # normalize

    n = x.shape[0]
    n_train = int(np.ceil(n * prop_vec[0]))
    n_test = int(np.ceil(n * prop_vec[1]))
    n_val = n - n_train - n_test

    assert np.amin([n_train, n_test, n_val]) >= 1   

    if shuffle:
        rng = np.random.default_rng(seed)
        pi = rng.permutation(n)
    else:
        pi = xrange(0, n)

    pi_train = pi[0:n_train]
    pi_test = pi[n_train:n_train+n_test]
    pi_val = pi[n_train+n_test:n]

    train_x = x[pi_train]
    train_y = y[pi_train]

    test_x = x[pi_test]
    test_y = y[pi_test]

    val_x = x[pi_val]
    val_y = y[pi_val]  
    
    return train_x, train_y, test_x, test_y, val_x, val_y



def print_array_hist(x, label=None):
    assert len(x.shape) <= 1 or x.shape[1] == 1

    if label is not None:
        print('--- {} ---'.format(label))
    for v in np.unique(x):
        print('{}: {}'.format(v, np.sum(x == v)))



def print_array_basic_stats(x, label=None):
    assert len(x.shape) <= 1 or x.shape[1] == 1

    if label is not None:
        print('--- {} ---'.format(label))

    print('min: {:.2f}'.format(np.amin(x)))
    print('max: {:.2f}'.format(np.max(x)))
    print('mean (+- std): {:.2f} (+- {:.2f})'.format(np.mean(x), np.std(x)))      



