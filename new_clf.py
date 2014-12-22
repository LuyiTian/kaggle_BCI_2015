'''
new_clf.py
__author__:Luyi Tian
'''

from datamodel import COLUMNS,LOCATION
import pandas as pd
import numpy as np
import os
import random
try:
    NAME = os.environ["COMPUTERNAME"]
except KeyError as e:
    NAME = os.environ['USER']

####################get coefficent matrix
def _spread_fun(x):
    return 1./(1.+10*x)**3
def _get_xy_locs():
    locations = pd.read_csv(os.path.join(LOCATION[NAME], 'ChannelsLocation.csv'))
    theta = np.radians(locations.Phi.values)
    r = locations.Radius.values
    _x = np.atleast_2d(r*np.cos(theta)).T
    _y = np.atleast_2d(r*np.sin(theta)).T
    pos = np.hstack((_x,_y))
    return pos
def _d(a,b):
    return np.sum((a-b)**2)
def get_dist():
    '''
    '''
    res = []
    pos = _get_xy_locs()
    for i in range(len(pos)):
        tmp = []
        for j in range(len(pos)):
            tmp.append(_spread_fun(_d(pos[i],pos[j])))
        res.append(np.atleast_2d(tmp).T)
    return res

def _get_TF_index(labels):
    T = [ith for ith,da in enumerate(labels) if da == 1.]
    F = [ith for ith,da in enumerate(labels) if da == 0.]
    return T,F
def _init_filter(lens):
    return np.random.rand(lens,1)
def _normal(a):
    return a/np.sum(np.abs(a))
def train_filter(X,labels,iters = 10000,acc_r = 0.1,bias = -0.3,lern_r = 0.1):
    '''
    '''
    dat_w = np.random.rand(iters) - bias
    if_acc = np.random.rand(iters)
    coeff_list = get_dist()
    sensor_num = X[0].shape[1]
    assert sensor_num == len(coeff_list), 'sensor num doesnot match between training data and ChannelsLocation'
    flt = _init_filter(sensor_num)
    flt = _normal(flt)
    T,F = _get_TF_index(labels)
    scores = [0]
    best_score = 0.
    best_flt = None
    for i in range(iters):
        coeff = random.choice(coeff_list)
        tmp_flt = lern_r*_normal(coeff*dat_w[i])+flt
        tmp_flt = _normal(tmp_flt)
        sig = [np.dot(da,tmp_flt) for da in X]
        tmp_score = _d(np.array([da for ith,da in enumerate(sig) if ith in T]).mean(0),\
             np.array([da for ith,da in enumerate(sig) if ith in F]).mean(0))
        print i,tmp_score
        if tmp_score>best_score:
            best_score = tmp_score
            best_flt = tmp_flt
        if tmp_score>scores[-1]:
            scores.append(tmp_score)
            flt = tmp_flt
        elif if_acc[i]<acc_r:
            scores.append(tmp_score)
            flt = tmp_flt
        else:
            scores.append(scores[-1])
    return best_score,best_flt,scores





if __name__ == '__main__':
    import datamodel
    global_setting = {'source': 'npy'}
    ##data object
    the_data = datamodel.EEGData()
    the_data.get_all(**global_setting)
    best_score,best_flt,scores = train_filter(the_data.train_signal_data, the_data.train_labels)
    import cPickle as pickle 
    pickle.dump((best_score,best_flt,scores),open('result.pkc','wb'))