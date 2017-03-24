#!/usr/bin/python

import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MaxAbsScaler

from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'long_term_incentive',
                 'deferred_income', 'deferral_payments', 'loan_advances',
                 'other', 'expenses', 'director_fees', 'total_payments',
                 'exercised_stock_options', 'restricted_stock',
                 'restricted_stock_deferred', 'total_stock_value',
                 'from_poi_to_this_person', 'shared_receipt_with_poi',
                 'to_messages', 'from_this_person_to_poi', 'from_messages',
                 'from_this_person_to_poi_p']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
#Transforming the dataset to pandas dataframe for easier handling
data_df = pd.DataFrame.from_dict(data_dict, orient='index')
#Replaces 'NaN's with '0' for compatibility with sklearn
data_df.replace(to_replace="NaN", value=0, inplace=True)
#Removes email field. It cannot be used somehow.
data_df = data_df.drop("email_address", axis=1)
#Rearranging the columns
cols = [
    'poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income',
    'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees',
    'total_payments', 'exercised_stock_options', 'restricted_stock',
    'restricted_stock_deferred', 'total_stock_value',
    'from_poi_to_this_person', 'shared_receipt_with_poi', 'to_messages',
    'from_this_person_to_poi', 'from_messages'
]
data_df = data_df[cols]

### Task 2: Remove outliers
#Dropping TOTAL
data_df.drop("TOTAL", inplace=True)
#Correcting erroneous records
data_df.loc["BELFER ROBERT", :] = [False, 0, 0, 0, 0, -102500, 3285, 0, 0,
                                   102500, 3285, -44093, 0, 44093, 0, 0, 0, 0,
                                   0, 0]
data_df.loc["BHATNAGAR SANJAY", :] = [False, 0, 0, 0, 0, 0, 137864, 0, 0, 0,
                                      137864, -2604490, 15456290, 2604490,
                                      15456290, 0, 463, 523, 1, 29]
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#Creating proportion of mails to pois to all sent mails
data_df.loc[:,"from_this_person_to_poi_p"] = data_df.loc[:,
           "from_this_person_to_poi"] / data_df.loc[:,"from_messages"]
#This division with zero created some NaNs.
data_df.replace(to_replace=np.NaN, value=0, inplace=True)
#Converting the DataFrame back to dictionary
my_dataset = data_df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
X = data_df.copy()
#Removing the poi labels and put them in a separate array, transforming it
#from True / False to 0 / 1
y = X.pop("poi").astype(int)

#Linear Support Vector Classifier
clf = Pipeline([('reduce_dim', SelectKBest(k=9)),
                ('classify', LinearSVC(C=1.0,  class_weight=None, dual=True,
                                       fit_intercept=True, intercept_scaling=1,
                                       loss='squared_hinge', max_iter=1000,
                                       multi_class='ovr', penalty='l2',
                                       random_state=42, tol=0.0001,
                                       verbose=0))])
clf.fit(X, y)

#Support Vector Classifier
clf = Pipeline([('scale', MaxAbsScaler(copy=True)),
                ('reduce_dim', SelectKBest(k=2)),
                ('classify', SVC(C=1, cache_size=200, class_weight=None,
                                 coef0=0.0, decision_function_shape=None,
                                 degree=3, gamma='auto', kernel='rbf',
                                 max_iter=-1, probability=False,
                                 random_state=42, shrinking=True, tol=0.001,
                                 verbose=False))])
clf.fit(X, y)

#Nearest Neighbors
clf = Pipeline([('reduce_dim', SelectKBest(k=5)),
                ('classify', KNeighborsClassifier(algorithm='auto',
                                                  leaf_size=30,
                                                  metric='minkowski',
                                                  metric_params=None, n_jobs=1,
                                                  n_neighbors=1, p=2,
                                                  weights='uniform'))])
clf.fit(X, y)

#Random Forest
clf = Pipeline([('reduce_dim', PCA(copy=True, iterated_power='auto',
                                   n_components=18, random_state=42,
                                   svd_solver='auto', tol=0.0, whiten=False)),
                ('classify', RandomForestClassifier(bootstrap=True,
                                                    class_weight=None,
                                                    criterion='gini',
                                                    max_depth=None,
                                                    max_features='auto',
                                                    max_leaf_nodes=None,
                                                    n_estimators=1, n_jobs=1,
                                                    oob_score=False,
                                                    random_state=42,
                                                    verbose=0,
                                                    warm_start=False))])
clf.fit(X, y)

#AdaBoost 
clf = Pipeline([('reduce_dim', SelectKBest(k=13)),
                ('classify', AdaBoostClassifier(algorithm='SAMME.R',
                                                base_estimator=None,
                                                learning_rate=1.0,
                                                n_estimators=10,
                                                random_state=42))])
clf.fit(X, y)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = Pipeline([('scale', MaxAbsScaler(copy=True)),
                ('reduce_dim', PCA(copy=True, iterated_power='auto',
                                   n_components=2, random_state=42,
                                   svd_solver='auto', tol=0.0, whiten=False)),
                ('classify', NearestCentroid(metric='manhattan',
                                             shrink_threshold=None))])

clf.fit(X, y)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)