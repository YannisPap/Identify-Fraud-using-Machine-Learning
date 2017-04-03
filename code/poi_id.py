#!/usr/bin/python

import sys
sys.path.append("./code/")

#from matplotlib.colors import ListedColormap
#import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC

from tester import test_classifier, dump_classifier_and_data
import warnings

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

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
print("\nReading data\n")
print("--------------------------------------------------------------------\n")
with open("../dataset/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
#Transforming the dataset to pandas dataframe for easier handling
data_df = pd.DataFrame.from_dict(data_dict, orient='index')
data_df.replace(to_replace='NaN', value=np.nan, inplace=True)
print("Initial dataset's dimensions: " + str(data_df.shape))
print("--------------------------------------------------------------------\n")
print()

#Replaces 'NaN's with '0' for compatibility with sklearn
print("Number of values per feature:")
print(data_df.count().sort_values())
print("--------------------------------------------------------------------\n")
print()

#Removes email field. It cannot be used somehow.
print("Removing 'email_address' field\n")
print("--------------------------------------------------------------------\n")
data_df = data_df.drop("email_address", axis=1)
print("Removing 'LOCKHART EUGENE E' because all features are empty\n")
print("--------------------------------------------------------------------\n")
data_df = data_df.drop(["LOCKHART EUGENE E"], axis=0)
#Rearranging the columns
cols = [
    'poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income',
    'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees',
    'total_payments', 'exercised_stock_options', 'restricted_stock',
    'restricted_stock_deferred', 'total_stock_value',
    'from_poi_to_this_person', 'shared_receipt_with_poi', 'to_messages',
    'from_this_person_to_poi', 'from_messages'
]
print("Rearranging features to: " + str(cols) + "\n")
print("--------------------------------------------------------------------\n")
data_df = data_df[cols]
print("Imputting missing values with '0'\n")
print("--------------------------------------------------------------------\n")
data_df.replace(to_replace="NaN", value=0, inplace=True)
### Task 2: Remove outliers
#Dropping TOTAL 
data_df.drop("TOTAL", inplace=True)

#Correcting erroneous records
print("Correcting errors on 'BELFER ROBERT' and 'BHATNAGAR SANJAY'\n")
print("--------------------------------------------------------------------\n")
data_df.loc["BELFER ROBERT", :] = [False, 0, 0, 0, 0, -102500, 3285, 0, 0,
                                   102500, 3285, -44093, 0, 44093, 0, 0, 0, 0,
                                   0, 0]
data_df.loc["BHATNAGAR SANJAY", :] = [False, 0, 0, 0, 0, 0, 137864, 0, 0, 0,
                                      137864, -2604490, 15456290, 2604490,
                                      15456290, 0, 463, 523, 1, 29]
print("Removing 'THE TRAVEL AGENCY IN THE PARK' as non employee\n")
print("--------------------------------------------------------------------\n")
data_df = data_df.drop(["THE TRAVEL AGENCY IN THE PARK"], axis=0)
print("Final shape of the dataset: " + str(data_df.shape))
print("--------------------------------------------------------------------\n")
print("Number of POIs in the dataset:\n" + str(data_df.loc[:, "poi"].value_counts()))
print("--------------------------------------------------------------------\n")
print("Initial classification score:\n")
features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')

test_classifier(LinearSVC(random_state=42), data, features)
print("--------------------------------------------------------------------\n")
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#Creating proportion of mails to pois to all sent mails
print("Creating 'proportion features'\n")
print("--------------------------------------------------------------------\n")
data = data_df.copy()
data.loc[:, "salary_p"] = data.loc[:, "salary"]/data.loc[:, "total_payments"]
data.loc[:, "deferral_payments_p"] = data.loc[:, "deferral_payments"]/data.loc[:, "total_payments"]
data.loc[:, "loan_advances_p"] = data.loc[:, "loan_advances"]/data.loc[:, "total_payments"]
data.loc[:, "bonus_p"] = data.loc[:, "bonus"]/data.loc[:, "total_payments"]
data.loc[:, "deferred_income_p"] = data.loc[:, "deferred_income"]/data.loc[:, "total_payments"]
data.loc[:, "expenses_p"] = data.loc[:, "expenses"]/data.loc[:, "total_payments"]
data.loc[:, "other_p"] = data.loc[:, "other"]/data.loc[:, "total_payments"]
data.loc[:, "long_term_incentive_p"] = data.loc[:, "long_term_incentive"]/data.loc[:, "total_payments"]
data.loc[:, "director_fees_p"] = data.loc[:, "director_fees"]/data.loc[:, "total_payments"]

data.loc[:, "restricted_stock_deferred_p"] = data.loc[:, "restricted_stock_deferred"]/data.loc[:, "total_stock_value"]
data.loc[:, "exercised_stock_options_p"] = data.loc[:, "exercised_stock_options"]/data.loc[:, "total_stock_value"]
data.loc[:, "restricted_stock_p"] = data.loc[:, "restricted_stock"]/data.loc[:, "total_stock_value"]

data.loc[:, "from_poi_to_this_person_p"] = data.loc[:, "from_poi_to_this_person"]/data.loc[:, "to_messages"]
data.loc[:, "shared_receipt_with_poi_p"] = data.loc[:, "shared_receipt_with_poi"]/data.loc[:, "to_messages"]

data.loc[:, "from_this_person_to_poi_p"] = data.loc[:, "from_this_person_to_poi"]/data.loc[:, "from_messages"]
    
data.replace(to_replace=np.NaN, value=0, inplace=True)
data.replace(to_replace=np.inf, value=0, inplace=True)
data.replace(to_replace=-np.inf, value=0, inplace=True)

print("Evaluating new features importance\n")
def do_split(data):
    X = data.copy()
    #Removing the poi labels and put them in a separate array, transforming it
    #from True / False to 0 / 1
    y = X.pop("poi").astype(int)
    
    return X, y, 

def plot_importance(dataset):
    X, y = do_split(dataset)

    selector = SelectPercentile(percentile=100)
    a = selector.fit(X, y)

    plt.figure(figsize=(12,9))
    sns.barplot(y=X.columns, x=a.scores_)
    sns.plt.show()

plot_importance(data)
print("--------------------------------------------------------------------\n")

print("Replace original features with proportions that perform better\n")

#Adding the proportions
data_df.loc[:, "long_term_incentive_p"] = data_df.loc[:, "long_term_incentive"]/data_df.loc[:, "total_payments"]
data_df.loc[:, "restricted_stock_deferred_p"] = data_df.loc[:, "restricted_stock_deferred"]/data_df.loc[:, "total_stock_value"]
data_df.loc[:, "from_this_person_to_poi_p"] = data_df.loc[:, "from_this_person_to_poi"]/data_df.loc[:, "from_messages"]
#Removing the original values.
data_df.drop("long_term_incentive", axis=1)
data_df.drop("restricted_stock_deferred", axis=1)
data_df.drop("from_this_person_to_poi", axis=1)
#Correcting NaN and infinity values created by zero divisions
data_df.replace(to_replace=np.NaN, value=0, inplace=True)
data_df.replace(to_replace=np.inf, value=0, inplace=True)
data_df.replace(to_replace=-np.inf, value=0, inplace=True)

plot_importance(data_df)
print("--------------------------------------------------------------------\n")

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
pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', LinearSVC(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 21))

param_grid = [
    {
        'scale':[None, MaxAbsScaler(), StandardScaler()],
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
    },
    {
        'scale':[None, MaxAbsScaler(), StandardScaler()],
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)
print("Score after scalling and feature selection\n")
test_classifier(grid.best_estimator_, my_dataset, features)
print("-----------------------------------------------------------------------\n")

print("Testing different families of classifiers\n")
print("Support Vector Machines\n")

#Support Vector Classifier
pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', SVC(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 10))
C_VALUES = [0.1, 1, 10]


param_grid = [
    {
        'scale':[None, MaxAbsScaler()],
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_VALUES
    },
    {
        'scale':[None, MaxAbsScaler()],
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_VALUES
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)

features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')
test_classifier(grid.best_estimator_, data, features)
print("--------------------------------------------------------------------\n")

print("Nearest Neighbors\n")
#Nearest Neighbors
pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', KNeighborsClassifier())])

N_FEATURES_OPTIONS = list(range(2, 21))
N_NEIGHBORS = [1, 3, 5]

param_grid = [
    {
        'scale': [None, MaxAbsScaler()],
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__weights': ["uniform", "distance"],
        'classify__n_neighbors': N_NEIGHBORS,
        'classify__p':[1, 2]
    },
    {
        'scale': [None, MaxAbsScaler()],
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__weights': ["uniform", "distance"],
        'classify__n_neighbors': N_NEIGHBORS,
        'classify__p':[1, 2]
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)

test_classifier(grid.best_estimator_, data, features)
print("--------------------------------------------------------------------\n")

#Random Forest
print("Ensemble Methods - Averaging\n")

pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify',  RandomForestClassifier(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 21))
N_TREES = [1, 2, 3]

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_TREES,
        'classify__criterion': ["gini", "entropy"]
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_TREES,
        'classify__criterion': ["gini", "entropy"]
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)
test_classifier(grid.best_estimator_, data, features)
print("--------------------------------------------------------------------\n")

#AdaBoost 
print("Ensemble Methods - Boosting\n")
pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify',  AdaBoostClassifier(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 21))
N_ESTIMATORS = [1, 10, 100]

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_ESTIMATORS,
        'classify__algorithm': ['SAMME', 'SAMME.R']
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_ESTIMATORS,
        'classify__algorithm': ['SAMME', 'SAMME.R']
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)

test_classifier(grid.best_estimator_, data, features)
print("--------------------------------------------------------------------\n")


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print("Nearest Neighbors family has the best score. We have already evaluated \
      K-Nearest Neighbor. Now I will evaluate and tune Nearest Centroid \
      Classifier\n")
pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', NearestCentroid())])

N_FEATURES_OPTIONS = list(range(2, 5))

param_grid = [
    {
        'scale': [None, MaxAbsScaler()],
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__metric': ["euclidean", "manhattan"],
        'classify__shrink_threshold': [None, 0.1, 1, 10]
    },
    {
        'scale': [None, MaxAbsScaler()],
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__metric': ["euclidean", "manhattan"],
        'classify__shrink_threshold': [None, 0.1, 1, 10]
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)
clf = grid.best_estimator_

best_score = test_classifier(clf, data, features)
best_score
print("--------------------------------------------------------------------\n")

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print("Dumping files")
dump_classifier_and_data(clf, my_dataset, features_list)
print("--------------------------------------------------------------------\n")

print("Process Completed\n")
print("Best Classifier / Score:\n")
print(best_score)
print()
print("Graphical Representation:\n")
h = .02  #step size in the mesh

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

scale = MaxAbsScaler()
X_trans = scale.fit_transform(X)
pca = PCA(copy=True,
          iterated_power='auto',
          n_components=2,
          random_state=42,
          svd_solver='auto',
          tol=0.0,
          whiten=False)
X_trans = pca.fit_transform(X_trans)
y_trans = y.values

clf = NearestCentroid(metric='manhattan', shrink_threshold=None)
clf.fit(X_trans, y_trans)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_trans[:, 0].min() - 0.1, X_trans[:, 0].max() + 0.1
y_min, y_max = X_trans[:, 1].min() - 0.1, X_trans[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y_trans, cmap=cmap_bold, alpha=0.3)

# Legend Data
classes = ['POI','Non-POI']
class_colours = ['#0000FF', '#FF0000']
recs = []
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))

# Plot Legend ('loc' == position):
plt.legend(recs,classes,loc=4)

plt.title("NearestCentroid classifier with manhattan metric")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")

plt.show()