'''
get data
__author__:Luyi Tian
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.decomposition import FastICA
from scipy.signal import firwin, lfilter
#from test_xDAWN import xDAWN

#hard coded parameters
LOCATION = {
    "JIAMING-SURFACE": os.path.normpath("C:/Kaggle/BCI_NER_2015"),
    "luyi": os.path.normpath("/Users/luyi/CTI_Challenge")

}
COLUMNS = ['Fp1', 'Fp2',\
           'AF7', 'AF3', 'AF4', 'AF8',\
           'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',\
           'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',\
           'FT8', 'T7',\
           'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',\
           'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',\
           'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
           'PO7', 'POz', 'P08', 'O1', 'O2']
###################

class EEGData:
    '''
    for data retriving, data cleaning and basic data transformation
    LOCATION/->
            train/      ->for all training data
            test/       ->for all testing data
            tmp_data/   ->store processed data 
            ./TrainLabels.csv
            ./SampleSubmission.csv
    ##########

    TODO
    '''
    def __init__(self):
        try:
            self.pc_name = os.environ["COMPUTERNAME"]
        except KeyError as e:
            self.pc_name = os.environ['USER']
        self.Df_train_label = pd.read_csv(os.path.join(LOCATION[self.pc_name], 'TrainLabels.csv'))
        self.Df_test_label = pd.read_csv(os.path.join(LOCATION[self.pc_name], 'SampleSubmission.csv'))
        ##get all training file 
        self.train_name = np.unique([it[:-6] for it in self.Df_train_label.IdFeedBack.values])
        self.train_file_path = [os.path.join(LOCATION[self.pc_name],'train/Data_{0}.csv'.format(it)) for it in self.train_name]
        ##get all testing file
        self.test_name = np.unique([it[:-6] for it in np.unique(self.Df_test_label.IdFeedBack.values)])
        self.test_file_path = [os.path.join(LOCATION[self.pc_name],'test/Data_{0}.csv'.format(it)) for it in self.test_name]
        ##np.unique will sort the data but it wont change the order in this case
        ##store training data
        self.train_signal_data = []
        self.train_mixing = []
        self.train_metadata = []
        self.train_labels = self.Df_train_label.Prediction.values
        ##store testing data
        self.test_signal_data = []
        self.test_mixing = []
        self.test_metadata = []
    def get_data_from_df(self,**kwargs):
        '''
        get data from Dataframe
        @param
            bound: (lo,up)data index from p-lo to p+up,p in the timepoint
                    where feedbackevents onset
        '''
        subject = float(kwargs['_f_name'][1:3])
        session = float(kwargs['_f_name'][-2:])
        df = kwargs['_Dataframe']
        
        event_onset = df[df.FeedBackEvent != 0].index.values
        df = df.loc[:,COLUMNS]
        _lo,_up = kwargs['bound']
        if kwargs.has_key('_filter'):
            df = df.apply(lambda x: lfilter(kwargs['_filter'], 1.0, x))
        if kwargs['mode'] == 'ICA':
            ica = FastICA(n_components = kwargs['n_components'],random_state = kwargs['random_state'])
            for ith, p in enumerate(event_onset):
                m = df[int(p)+_lo-80:int(p)+_lo+20]
                sig = df[int(p)+_lo:int(p)+_up]-m.mean(0)
                ICs = ica.fit_transform(sig.values)
                if kwargs['_train']:
                    tmp = np.array([da>0 and 1. or -1 for da in ica.mixing_.T.mean(1)])
                    self.train_signal_data.append(ICs.T)
                    self.train_mixing.append(ica.mixing_.T)
                    self.train_metadata.append(np.array([subject,session,ith+1,p]))
                else :
                    tmp = np.array([da>0 and 1. or -1 for da in ica.mixing_.T.mean(1)])##TODO: better way?
                    self.test_signal_data.append(ICs.T)
                    self.test_mixing.append(ica.mixing_.T)
                    self.test_metadata.append(np.array([subject,session,ith+1,p]))
        else:
            raise NotImplementedError,"mode mast be 'ICA'."

    def get_all(self,**kwargs):
        '''
        TODO
        '''
        if kwargs['source'] == 'file_to_npy' or kwargs['source'] == 'file':
            if kwargs['low_pass']:
                N=15
                Fs=200
                h=firwin(numtaps=N, cutoff=[1,30], nyq=Fs/2, pass_zero=False)
                kwargs['_filter'] = h
            #get training data
            kwargs['_train'] = True
            for f_l,name in zip(self.train_file_path,self.train_name):
                print 'training data:',name
                df = pd.read_csv(f_l)
                kwargs['_Dataframe'] = df
                kwargs['_f_name'] = name
                self.get_data_from_df(**kwargs)
            #get testing data
            kwargs['_train'] = False
            for f_l,name in zip(self.test_file_path,self.test_name):
                print 'testing data:',name
                df = pd.read_csv(f_l)
                kwargs['_Dataframe'] = df
                kwargs['_f_name'] = name
                self.get_data_from_df(**kwargs)
            if kwargs['source'] == 'file_to_npy':
                np.save(os.path.join(LOCATION[self.pc_name],"tmp_data/training_signal.npy"),np.array(self.train_signal_data))
                np.save(os.path.join(LOCATION[self.pc_name],"tmp_data/training_metadata.npy"),np.array(self.train_metadata)) 
                np.save(os.path.join(LOCATION[self.pc_name],"tmp_data/training_mixing.npy"),np.array(self.train_mixing))
                np.save(os.path.join(LOCATION[self.pc_name],"tmp_data/testing_signal.npy"),np.array(self.test_signal_data))
                np.save(os.path.join(LOCATION[self.pc_name],"tmp_data/testing_metadata.npy"),np.array(self.test_metadata)) 
                np.save(os.path.join(LOCATION[self.pc_name],"tmp_data/testing_mixing.npy"),np.array(self.test_mixing))
        if kwargs['source'] == 'npy':
            self.train_signal_data = [da for da in np.load(os.path.join(LOCATION[self.pc_name],"tmp_data/training_signal.npy"))]
            self.train_metadata = np.load(os.path.join(LOCATION[self.pc_name],"tmp_data/training_metadata.npy")) 
            self.train_mixing = [da for da in np.load(os.path.join(LOCATION[self.pc_name],"tmp_data/training_mixing.npy"))]
            self.test_signal_data = [da for da in np.load(os.path.join(LOCATION[self.pc_name],"tmp_data/testing_signal.npy"))]
            self.test_metadata = np.load(os.path.join(LOCATION[self.pc_name],"tmp_data/testing_metadata.npy")) 
            self.test_mixing = [da for da in np.load(os.path.join(LOCATION[self.pc_name],"tmp_data/testing_mixing.npy"))]

    def get_train_subject(self):
        return list(set([it[0] for it in self.train_metadata]))
    def get_test_subject(self):
        return list(set([it[0] for it in self.test_metadata]))








if __name__ == '__main__':
    a = EEGData()
    all_arg = {'bound': (0,260),'mode': 'ICA', 'low_pass':True,'source': 'file_to_npy',\
                'n_components': 5, 'random_state': 10}

    a.get_all(**all_arg)


