'''
__author__:Luyi Tian
'''
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from scipy.linalg import qr,svd

def build_toe_mat(N_t,N_e,tau):
    '''
    return: D(N_t,N_e)
    D is the Toeplitz matrix whose first column is defined 
    such that D[tau,0] = 1 
    '''
    D = np.zeros((tau[0],N_e))
    for t in range(len(tau)-1):
        D = np.vstack((D,np.diag(np.ones(N_e))))
        D = np.vstack((D,np.zeros((tau[t+1]-tau[t]-N_e,N_e))))
    D = np.vstack((D,np.diag(np.ones(N_e))))
    D = np.vstack((D,np.zeros((N_t-tau[-1]-N_e,N_e))))
    return D

def xDAWN(X,tau,N_e,remain = 5):
    '''
    xDAWN spatial filter for enhancing event-related potentials.
    
    xDAWN tries to construct spatial filters such that the 
    signal-to-signal plus noise ratio is maximized. This spatial filter is 
    particularly suited for paradigms where classification is based on 
    event-related potentials.
    
    For more details on xDAWN, please refer to 
    http://www.icp.inpg.fr/~rivetber/Publications/references/Rivet2009a.pdf

    this code is inspired by 'xdawn.py' in pySPACE:
    https://github.com/pyspace/pyspace/blob/master/pySPACE/missions/nodes/spatial_filtering/xdawn.py
    ##################
    use the same notations as in the paper linked above (../River2009a.pdf)
    N_t:the number of temporal samples. (over 100,000 per session)
    N_s:the number of sensors. (56 EEG sensors)
    @param:
        X: input EEG signal (N_t,N_s)
        tau: index list where stimulus onset
        N_e: number of temporal samples of the ERP (<1.3sec)
    return:

    '''
    N_t,N_s= X.shape
    D = build_toe_mat(N_t,N_e,tau)#construct Toeplitz matrix
    #print X.shape
    Qx, Rx = qr(X, overwrite_a = True, mode='economic')
    Qd, Rd = qr(D, overwrite_a = True, mode='economic')  
    Phi,Lambda,Psi = svd(np.dot(Qd.T,Qx),full_matrices = True)
    Psi = Psi.T
    SNR = []
    U = None
    A = None
    for i in range(remain):
        ui = np.dot(np.linalg.inv(Rx),Psi[:,i])
        ai = np.dot(np.dot(np.linalg.inv(Rd),Phi[:,i]),Lambda[i])
        if U == None:
            U = np.atleast_2d(ui).T
        else:
            U = np.hstack((U,np.atleast_2d(ui).T))
        if A == None:
            A = np.atleast_2d(ai)
        else:
            A = np.vstack((A,np.atleast_2d(ai)))
        tmp_a = np.dot(D, ai.T)
        tmp_b = np.dot(X, ui)
        #print np.dot(tmp_a.T,tmp_a)/np.dot(tmp_b.T,tmp_b)
    return U,A


if __name__ == "__main__":
    import cPickle as pickle 
    tau,X = pickle.load(open('S02_Sess03_FB001-FB060.pkl'))
    U,A = xDAWN(X,tau,260)
    print X.shape,U.shape
    sig = np.dot(X,U)
    sig = sig.T
    import matplotlib.pylab as plt
    fig, ax_ = plt.subplots(nrows=10, sharex=True)
    for aaa,it in zip(ax_, sig):
        aaa.plot([i*5 for i in range(260)],it[tau[1]:tau[1]+260])#[i*5 for i in range(200)]
    plt.show()