'''
classifiers
__author__:Luyi Tian
'''
import datamodel
import get_feature
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
import numpy as np
import random
def get_auc_on_known_data(clf,training_data,training_label,testing_data,testing_label):
    clf.fit(training_data,training_label)
    preds = clf.predict_proba(testing_data)[:,1]
    auc = roc_auc_score(testing_label, preds)
    return auc
def get_subset(data,label,subject_list):
    test_list = [ith for ith,da in enumerate(data) if da[0] in subject_list]
    train_list = [ith for ith in range(data.shape[0]) if ith not in test_list]
    training_data = data[train_list]
    training_label = label[train_list]
    testing_data = data[test_list]
    testing_label = label[test_list]
    return training_data,training_label,testing_data,testing_label


if __name__ == '__main__':
    the_data = datamodel.EEGData()
    all_arg = {'bound': (-200,300),'mode': 'ICA', 'low_pass':True,'source': 'npy',\
                'n_components': 10, 'random_state': 10}
    the_data.get_all(**all_arg)
    new_train = []
    the_freq = get_feature.DataFeatures(N_t = 500)
    for ith,ICs in enumerate(the_data.train_signal_data):
        ICs = ICs.tolist()
        tmp = the_data.train_mixing[ith] /the_data.train_mixing[ith].std(0)
        tmp = tmp.T
        ICs = [a for a,b in zip(ICs,tmp[-1]) if -2.<b<2.]#remove EOG
        ICs.sort(key = lambda x:wrapper(the_freq,x))
        ICs = np.array(ICs)[:5,200:400:2]
        #ICs = [sum(ICs[:,i]) for i in range(ICs.shape[1])]
        new_train.append(np.append(the_data.train_metadata[ith][0],np.ravel(ICs)))#np.append(meta,np.ravel(ICs)))
    new_train = np.array(new_train)
    print len(new_train)
    subjects = the_data.get_train_subject()
    print subjects
    tmp = random.sample(subjects,4)
    print tmp
    training_data,training_label,testing_data,testing_label = get_subset(new_train,the_data.train_labels,tmp)
    clf = ensemble.GradientBoostingClassifier(n_estimators=600, learning_rate=0.05, max_features=0.25)
    print get_auc_on_known_data(clf,training_data,training_label,testing_data,testing_label)

