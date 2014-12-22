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

def simple_cv(clf,data,label,sub_list,sub_num = 4,iter = 10):
    auc_list = []
    for ith in range(iter):
        tmp = random.sample(sub_list,sub_num)
        training_data,training_label,testing_data,testing_label = get_subset(data,label,tmp)
        auc = get_auc_on_known_data(clf,training_data,training_label,testing_data,testing_label)
        print auc
        auc_list.append(auc)
    return auc_list


if __name__ == '__main__':
    from sklearn.lda import LDA
    the_data = datamodel.EEGData()
    all_arg = {'bound': (-100,300),'mode': 'ICA', 'low_pass':True,'source': 'npy',\
                'n_components': 15, 'random_state': 10}
    the_data.get_all(**all_arg)
    sub = the_data.get_train_subject()
    subset_l = random.sample(sub,4)
    W_mat = 100*(np.random.rand(100,3))
    the_feature = get_feature.DataFeatures(N_t = 400)
    #clf = ensemble.GradientBoostingClassifier(n_estimators=600, learning_rate=0.05, max_features=0.25)
    clf = LDA()
    res_log = open('_'.join([str(int(it)) for it in subset_l]),'w+')
    for W in W_mat:
        print W
        new_train = []
        for ith,ICs in enumerate(the_data.train_signal_data):
            ICs = ICs.tolist()
            ICs = the_feature.remove_EOG(ICs,the_data.train_mixing[ith])
            the_feature.reorder_IC_by_features(ICs,W)
            ICs = the_feature.dim_reduction(ICs)
            new_train.append(np.append(the_data.train_metadata[ith],np.ravel(ICs)))
        new_train = np.array(new_train)
        training_data,training_label,testing_data,testing_label = get_subset(new_train,the_data.train_labels,subset_l)
        auc = get_auc_on_known_data(clf,training_data,training_label,testing_data,testing_label)
        print auc
        res_log.write(','.join([str(it) for it in W])+','+str(auc)+'\n')
    res_log.close()


