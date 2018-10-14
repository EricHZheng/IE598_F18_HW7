#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 00:55:08 2018

@author: ericzheng
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import timeit

#part1
#import wine dataset 
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

#train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

#
#forest = RandomForestClassifier()
#forest.get_params()

score_df = pd.DataFrame()
size = [100, 150, 200, 250, 300, 350, 400]
for i in size:
    start = timeit.default_timer()
    forest = RandomForestClassifier(n_estimators=i, criterion='gini', random_state=1, n_jobs=1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    cvscores_forest = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=1)
    out_sample_score = metrics.accuracy_score(y_test_pred, y_test)
    mean = np.mean(cvscores_forest)
    std = np.std(cvscores_forest)
    stop = timeit.default_timer()
    time = stop-start
    array = {f: s for f, s in zip()}
    array['n_estimator'] = i
    array['computation_time'] = time
    array['cvscores_mean_inSample'] = mean
    array['cvscores_std_inSample'] = std
    array['score_outSample'] = out_sample_score
    score_df=score_df.append(array, ignore_index=True)
    print("n_estimator = {}".format(i))
    print("cvscores:")
    print(cvscores_forest)
score_df = score_df.set_index('n_estimator')[['computation_time','cvscores_mean_inSample', 'cvscores_std_inSample', 'score_outSample']]
print(score_df)
print('\n')

#part2
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=300,
random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Hao Zheng")
print("My NetID is: haoz7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")