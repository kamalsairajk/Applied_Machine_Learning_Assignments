# -*- coding: utf-8 -*-
""" CIS4930/6930 Applied ML --- utils.py
"""


import json
import re
import os
import time


import numpy as np

def zscore_normalize(x,  features_to_standardize):
    assert x.shape[1] == len(features_to_standardize)

    z = x.copy()
    for i, standardize in enumerate(features_to_standardize):
        if standardize:
            zm = np.mean(z[:,i])
            z[:,i] = z[:,i] - zm
            zstd = np.std(z[:,i])
            z[:,i] = z[:,i] / zstd
    return z
    
def minmax_normalize(x,  features_to_normalize):
    assert x.shape[1] == len(features_to_normalize)

    z = x.copy()
    for i, normalize in enumerate(features_to_normalize):
        if normalize:
            zmin = np.amin(z[:,i])
            z[:,i] = z[:,i] - zmin
            zmax = np.amax(z[:,i])
            z[:,i] = z[:,i] / zmax
    return z
    
def one_hot_encode(x,  features_to_encode, col_names=None):
    assert x.shape[1] == len(features_to_encode)

    features_names = None if col_names is None else []

    z = None
    for i, encode in enumerate(features_to_encode):
        xi = x[:,i]
        
        if encode:
            xiint = xi.astype(int)
            nv = np.max(xiint) + 1
            oh = np.eye(nv)[xiint]
            
            if col_names is not None:
                for j in range(0, oh.shape[1]):
                    features_names.append('{}_{}'.format(col_names[i], j))           
        else:
            oh = xi.copy().reshape(-1, 1)
            if col_names is not None:
                features_names.append(col_names[i])   
            
        if z is None:
            z = oh
        else:
            z = np.concatenate((z, oh), axis=1)
            
    if col_names is None:  
        return z
    else:
        return z, features_names


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



def load_preproc_adult(prop_vec=[14, 3, 3], seed=None):
    import pandas as pd
    
    # Use pandas to load the data from compressed CSV
    df1 = pd.read_csv('../data/adult.data.gz', compression='gzip', header=0, na_values='?', sep=' *, *', skipinitialspace=True, engine='python')
    df2 = pd.read_csv('../data/adult.test.gz', compression='gzip', header=0, na_values='?',sep=' *, *', skipinitialspace=True, engine='python')
    
    # Check that we loaded the data as expected
    df1_expected_shape = (32561,15)
    df2_expected_shape = (16281,15)

    assert df1.shape == df1_expected_shape, 'Unexpected shape of df1!'
    assert df2.shape == df2_expected_shape, 'Unexpected shape of df2!'
    
    # Merge df1 and df2 for pre-processing and cleaning
    dfraw = df1.append(df2, ignore_index=True, sort=False)
    assert dfraw.shape == (df1_expected_shape[0]+df2_expected_shape[0],15)
    
    # Let's start cleaning up the data
    df = dfraw

    # Let's get rid of the record weights 'fnlwgt' and 'education' (which contains same information as 'education-num')
    if 'fnlwgt' in df.columns:
        df = df.drop('fnlwgt', axis=1)
    if 'education' in df.columns:
        df = df.drop('education', axis=1)

    # remove rows with NaN
    df = df.dropna(axis=0)
    
    # Now we need to recode the categorical attributes
    workclass_list = ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']

    marital_status_list = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']

    occupation_list = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving' ]
        
    relationship_list = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']

    race_list =['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White' ]

    sex_list = ['Male', 'Female']

    native_country_list = ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
            'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
            'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary',
            'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',
            'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland',
            'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand',
            'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']

    income_list = ['<=50K', '>50K']

    cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
    cat_lists = [workclass_list, marital_status_list, occupation_list, relationship_list, race_list, sex_list, native_country_list, income_list]

    # first make all the numerical cols integers
    is_cat = []
    for col_name in df.columns:
        cat = True
        if not col_name in cat_cols:
            df[col_name] = df[col_name].astype(float)
            cat = False
        is_cat.append(cat)

    # encode categorical
    for i, col_name in enumerate(cat_cols):
        df[col_name] = df[col_name].apply(lambda v: cat_lists[i].index(v))
        df[col_name] = df[col_name].astype(int)
        
    # finally, make education-num 0 based and integer
    df['education-num'] = (df['education-num']-1).astype(int)
        
        
    is_cat = [True if col_name in cat_cols else False for col_name in df.columns]
    is_cat[2] = False # education num is considered numerical for us here!
    col_names = df.columns
    col_names = [c for c in df.columns]
    
    df_expected_shape = (45222, 13)
    assert df.shape == df_expected_shape
    
    # grab all the data as a numpy array
    all_xy = np.asarray(df, dtype='float64')
    assert all_xy.shape[1] == 13

    label_col_idx = all_xy.shape[1]-1
    features_col_idx = range(0, label_col_idx)

    feature_names = col_names[0:label_col_idx]
    
    # separate features from the label
    all_x = all_xy[:,features_col_idx]
    all_y = all_xy[:,label_col_idx]

    is_cat_features = is_cat[0:len(is_cat)-1]
    
    stdb = [not cat for i, cat in enumerate(is_cat_features)]
    all_x_zscore = all_x.copy()
    all_x_zscore = zscore_normalize(all_x_zscore, stdb)
    
    all_x_zscore_onehot = all_x_zscore.copy()
    all_x_zscore_onehot, features_names = one_hot_encode(all_x_zscore_onehot, is_cat_features, col_names)
    
    features_x = all_x_zscore_onehot
    assert features_x.shape == (45222, 88)
    
    train_x, train_y, test_x, test_y, val_x, val_y = train_test_val_split(features_x, all_y, prop_vec, shuffle=True, seed=seed)
    
    return train_x, train_y, test_x, test_y, val_x, val_y, features_names, income_list
