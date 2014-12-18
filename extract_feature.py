import matplotlib.mlab as pym
import numpy as np

class FreqFeatures:
    def __init__(self,N_t = 500,Fs = 200,NFFT = 64):
        pass




if __name__ == "__main__":
    import cPickle as pickle 
    ICs = pickle.load(open('test_data/S02_Sess03_FB010_8ICs_R20.pkl'))
    Pow, freqs,bins = pym.specgram(ICs[-4],noverlap = 32,NFFT=64, Fs=200,)
    print len(freqs),Pow.shape