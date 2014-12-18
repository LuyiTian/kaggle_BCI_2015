'''
__author__:Luyi Tian
'''

import matplotlib.pyplot as plt
import matplotlib.mlab as plm
import numpy as np

class DataFeatures:
    '''
    TODO
    '''
    def __init__(self,N_t,Fs = 200,NFFT = 64):
        self.N_t = N_t 
        self.Fs = Fs 
        self.NFFT = NFFT
        self.Pow = None
        self.bins = None
        self.freqs = None
    def specg(self,x):
        self.Pow, self.freqs,self.bins = plm.specgram(x,noverlap = 32,NFFT=64, Fs=200,)
        self.Pow = np.log10(self.Pow)
    def get_mean_diff(self,freq_bound=(0,20),time_window=(0,0.7)):
        freq_index = [ith for ith,da in enumerate(self.freqs) if freq_bound[0]<=da<=freq_bound[1]]#TODO:better way to do this
        time_index = [ith for ith,da in enumerate(self.bins) if time_window[0]<=da<=time_window[1]]
        return self.Pow[:,time_index][freq_index].mean() - self.Pow[freq_index].mean()##TODO:self.Pow[freq_index,time_index] doesnot work. why?
    def current_density(self):
        raise NotImplementedError,'Do not support current density norm'
    def local_skewness(self):
        raise NotImplementedError,'Do not support averaged local skewness'
    def lambda_fiterror(self):
        '''
        lambda and FitError: deviation of a component's spectrum from
        a protoptypical 1/frequency curve 
        '''
        raise NotImplementedError,'Do not support lambda and FitError'
    def _wrapper(self,x):
        self.specg(x)
        return max(max(x),-min(x))#+self.get_mean_diff()
    def remove_EOG(self,ICs,mixingT,thr = 2.):
        '''

        '''
        tmp = mixingT /mixingT.std(0)
        tmp = tmp.T
        return [a for a,b in zip(ICs,tmp[-1]) if -2.3<b<2.3]#remove EOG
    def reorder_IC_by_features(self,ICs, W = None):
        ICs.sort(key = lambda x:self._wrapper(x))
    def dim_reduction(self,ICs,dim = 5,time_bound =(0,200)):
        '''

        '''
        return np.array(ICs)[:dim,time_bound[0]:time_bound[1]:2]



if __name__ == "__main__":
    import cPickle as pickle 
    ICs = pickle.load(open('test_data/S02_Sess03_FB010_8ICs_R20.pkl'))
    test = DataFeatures(N_t = 501)
    for IC in ICs:
        test.specg(IC)
        print test.get_mean_diff()
    fig, ax_ = plt.subplots(nrows=10, sharex=True)
    for aaa,it in zip(ax_, ICs):
        aaa.plot(it)
    plt.show()
    #print ICs.shape
    #Pow, freqs,bins,im = plt.specgram(ICs[3],noverlap = 32,NFFT=64, Fs=200,)
    #plt.show()
    #print bins