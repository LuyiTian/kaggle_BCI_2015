'''
main_prog.py
__author__:Luyi Tian
'''
import numpy as np
def data_processing(the_data,the_feature,train):
    new_data = []
    if train:
        for ith,ICs in enumerate(the_data.train_signal_data):
            ICs = ICs.tolist()
            #ICs = the_feature.remove_EOG(ICs,the_data.train_mixing[ith])
            the_feature.reorder_IC_by_features(ICs)
            ICs = the_feature.dim_reduction(ICs)
            new_data.append(np.append(the_data.train_metadata[ith],np.ravel(ICs)))
    else:
        for ith,ICs in enumerate(the_data.test_signal_data):
            ICs = ICs.tolist()
            #ICs = the_feature.remove_EOG(ICs,the_data.train_mixing[ith])
            the_feature.reorder_IC_by_features(ICs)
            ICs = the_feature.dim_reduction(ICs)
            new_data.append(np.append(the_data.test_metadata[ith],np.ravel(ICs)))
    return np.array(new_data)



if __name__ == '__main__':
    import datamodel
    import get_feature
    import classifiers
    from sklearn import ensemble
    global_setting = {'bound': (-100,300),'mode': 'ICA', 'low_pass':True,'source': 'npy',\
                'n_components': 5, 'random_state': 10}
    ##data object
    the_data = datamodel.EEGData()
    the_data.get_all(**global_setting)

    ##feature object
    the_feature = get_feature.DataFeatures(N_t = 300)

    ##get training data 
    training_data = data_processing(the_data,the_feature,train=True)
    print 'training_data:',len(training_data),len(training_data[0])

    ##get testing data 
    testing_data = data_processing(the_data,the_feature,train=False)
    print 'testing_data:',len(testing_data),len(testing_data[0])

    ##get clf
    clf = ensemble.GradientBoostingClassifier(n_estimators=600, learning_rate=0.05, max_features=0.25)


    ##simple cv
    print 'start cross validation'
    all_auc = classifiers.simple_cv(clf, training_data, the_data.train_labels, the_data.get_train_subject())
    print 'average auc:',sum(all_auc)/len(all_auc)
    ##train data
    print 'start training'
    clf.fit(training_data,the_data.train_labels)
    print 'finish training'
    ##predict, to csv
    preds = clf.predict_proba(testing_data)[:, 1]
    the_data.Df_test_label['Prediction'] = preds
    the_data.Df_test_label.to_csv('new_metadata_ICA_5d.csv', index=False)
