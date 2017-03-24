#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:38:10 2017

@author: yannis
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

with open("./dataset/final_project_dataset.pkl", "rb") as data_file:
    data_init = pickle.load(data_file)
data_df = pd.DataFrame.from_dict(data_init, orient='index')
data_df.replace(to_replace='NaN', value=np.nan, inplace=True)
data_df.replace(to_replace=np.nan, value=0, inplace=True)
data_df.email_address.replace(to_replace=0, value=np.nan, inplace=True)
data_df["to_poi_vs_from"] = data_df["from_this_person_to_poi"] / data_df[
    "from_messages"]
data_df["from_poi_vs_from"] = data_df["from_poi_to_this_person"] / data_df[
    "from_messages"]

X = data_df.drop("email_address", axis=1)
X.poi = X.poi.astype(int)
X.replace(to_replace=np.nan, value=0, inplace=True)
y = X.pop("poi")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

selector = SelectKBest(k=5)
a = selector.fit(X_train, y_train)

X_train = X_train[X.columns[selector.get_support()]]
X_test = X_test[X.columns[selector.get_support()]]
selected_features = X_train.columns

X_train = preprocessing.scale(X_train)
X_train = pd.DataFrame(X_train, columns=selected_features)


def find_optimal(min_bound, max_bound):
    while min_bound < max_bound:
        b = round((min_bound+max_bound)/2)
        print(min_bound, b, max_bound)
        tune_parameters = [{'n_estimators': [min_bound,b,max_bound]}]
        
        clf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), tune_parameters)
    
        clf.fit(X_train, y_train)
        best_param = clf.best_params_['n_estimators']
        
        if (best_param == min_bound):
            max_bound = b
        elif (best_param == max_bound):
            min_bound = b
        else:
            min_bound = round((b+min_bound)/2)
            max_bound = round((max_bound+b)/2)
            
    print(clf.best_params_)
    print(clf.best_score_)
    
find_optimal(1,1000)