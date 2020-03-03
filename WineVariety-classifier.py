from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse, sys

from datetime import datetime
from matplotlib import style

def main(samplesNumber, isKernel, isVerbose):
    filter = ['country','description','points','price','variety']
    dataset = pd.read_csv('WineReviews.csv', usecols=filter)

    dataset = preprocessWineDataset(dataset.copy())
    if(samplesNumber is not None):
        dataset = dataset.sample(samplesNumber)
        dataset = dataset.groupby('variety').filter(lambda x: len(x) >= 5) # k-folding needs at least 5 members

    dataset_dummy = pd.get_dummies(dataset.drop('variety', axis=1))

    X_train = dataset_dummy
    y_train = dataset['variety']

    if(not isKernel):
        hyperParams = {"estimator__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        clf, fitDuration = LinearKernel(X_train, y_train, hyperParams)        
    else:
        hyperParams = {"C":[0.01, 0.1, 1, 10, 100, 1000], "gamma":[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
        clf, fitDuration = RbfKernel(X_train, y_train, hyperParams)

    if(isVerbose):
        means = clf.cv_results_['mean_test_score']
        for mean, hyperParams in zip(means, clf.cv_results_['params']):
            print("{0:.0%} for {1}".format(mean, hyperParams))
        print('\nBest params: ' + str(clf.best_params_))
    else:
        print("Best accuracy: {:.0%}".format(means.max()))
    print("\nFit duration: " + str(fitDuration))
    

def RbfKernel(X_train, y_train, hyperParams):
    svm = SVC(decision_function_shape='ovo')
    clf = GridSearchCV(svm, hyperParams, n_jobs=10)

    print("Start fit...")

    start = datetime.now()
    clf.fit(X_train, y_train)
    end = datetime.now()

    print("End fit. Duration: {}".format(end - start))
    print()

    return clf, (end - start)

def LinearKernel(X_train, y_train, hyperParams):
    svm = OneVsOneClassifier(LinearSVC(dual=False), n_jobs=10)
    clf = GridSearchCV(svm, hyperParams, n_jobs=10)

    print("Start fit...")

    start = datetime.now()
    clf.fit(X_train, y_train)
    end = datetime.now()

    print("End fit!")
    print()

    return clf, (end - start)

def preprocessWineDataset(dataset):    
    dataset.drop_duplicates('description', inplace=True)
    dataset.drop("description", axis=1, inplace=True)
    dataset = dataset.dropna(how='any',axis=0)
    dataset = dataset.groupby('variety').filter(lambda x: len(x) > 200)
    dataset['variety'] = dataset['variety'].replace(['Weissburgunder'], 'Chardonnay')
    dataset['variety'] = dataset['variety'].replace(['Spatburgunder'], 'Pinot Noir')
    dataset['variety'] = dataset['variety'].replace(['Grauburgunder'], 'Pinot Gris')
    dataset['variety'] = dataset['variety'].replace(['Garnacha'], 'Grenache')
    dataset['variety'] = dataset['variety'].replace(['Pinot Nero'], 'Pinot Noir')
    dataset['variety'] = dataset['variety'].replace(['Alvarinho'], 'Albarino')       
    return dataset

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', action='store_true', help='use rbf kernel (default: linear)')
    parser.add_argument('-s', type=int, metavar='Samples',help='number of sample to use (default: all dataset samples)')
    parser.add_argument('-v', action='store_true', help='display all results for HyperParameters')
    args = parser.parse_args()

    main(args.s, args.k, args.v)

