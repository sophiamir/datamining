#                               %%%%%%%%%%%%%      DATA MINING     %%%%%%%%%%%%%
#                               %%%%%%%%%%%%%     FINAL PROJECT    %%%%%%%%%%%%%
#                               %%%%%%%%%%%%%  AUTHOR: Sophia Mir  %%%%%%%%%%%%%

# Email: smir@gwmail.gwu.edu
# June - 22 - 2022

"""DC Crimes Dataset"""
"""Primary data collected by The Metropolitan Police Department"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors as knn
import warnings
warnings.filterwarnings("ignore")

# importing the dataset and selecting the relevant columns
cdat = pd.read_csv('dc-crimes-search-results.csv')

# checking for missing data
print(cdat.isnull().sum())

# looking at the data types and changing them
print(cdat.info())
#cdat.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13]] = cdat.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13]].astype('category')
#print(cdat.info())

cdat = cdat.astype('category')
#cdat['CENSUS_TRACT'] = cdat['CENSUS_TRACT'].astype('float64')
#cdat['LONGITUDE'] = cdat['LONGITUDE'].astype('float64')
#cdat['LATITUDE'] = cdat['LATITUDE'].astype('float64')

# convert blank spaces
""" UNNECESSARY CODE """
#cdat['NEIGHBORHOOD_CLUSTER'] = cdat['NEIGHBORHOOD_CLUSTER'].isnull().fillna(np.NAN)
#cdat['DISTRICT'] = cdat['DISTRICT'].isnull().fillna(np.NAN)
#cdat['WARD'] = cdat['WARD'].isnull().fillna(np.NAN)

# dropping column with 45808 missing values
cdat.drop("BID", axis=1, inplace=True)

cdat2 = cdat.copy()

# dropping columns with repetitive data
cdat.drop(['offense-text', 'offensekey'], axis=1, inplace=True)

# dropping columns with unnecessary data
cdat.drop(['ucr-rank', 'START_DATE', 'END_DATE'], axis=1, inplace=True)

# imputing missing values
from sklearn.impute import SimpleImputer

fillin = SimpleImputer(missing_values=np.NAN, strategy="most_frequent")
cdat.loc[:, ['NEIGHBORHOOD_CLUSTER', 'CENSUS_TRACT', 'DISTRICT', 'WARD',
             'sector', 'PSA', 'BLOCK_GROUP', 'VOTING_PRECINCT',
             'ANC']] = fillin.fit_transform(cdat.loc[:, ['NEIGHBORHOOD_CLUSTER',
                                                         'CENSUS_TRACT', 'DISTRICT', 'WARD', 'sector',
                                                         'PSA', 'BLOCK_GROUP', 'VOTING_PRECINCT', 'ANC']])

#cdat.loc[:, ['NEIGHBORHOOD_CLUSTER', 'DISTRICT', 'WARD']] = cdat.loc[:, ['NEIGHBORHOOD_CLUSTER', 'DISTRICT', 'WARD']].astype('category')

cdat = cdat.astype('category')
#cdat['CENSUS_TRACT'] = cdat['CENSUS_TRACT'].astype('float64')
#cdat['LONGITUDE'] = cdat['LONGITUDE'].astype('float64')
#cdat['LATITUDE'] = cdat['LATITUDE'].astype('float64')

# looking at the data structure and summary statistics

print("No. of Rows:", cdat.shape[0])
print("No. of Columns:", cdat.shape[1])
print(cdat.isnull().sum())
print(cdat.info())

#print(cdat.isnull().sum())
#print(cdat.info())

# exploratory data analysis
# figuring out the characteristics of the data

print(cdat['SHIFT'].unique())
# 3 shifts: 'evening', 'day', 'midnight'
# 3 methods specified: 'gun', 'knife', 'other'
# 9 types of crimes: 'arson', 'assault w/dangerous weapon', 'theft f/auto', 'theft', 'homicide',
# 'sex abuse', 'robbery', 'motor vehicle theft', 'burglary'
# 8 wards
# 7 districts
# 46 neighborhood clusters
# data across 3 years: 2020, 2021, 2022
# 2 offense categories: 'property' and 'violent'


# creating a new column for the month value
# using code from (https://stackoverflow.com/questions/57896172/list-splice-for-each-row-in-column-of-dataframe)
# and (https://towardsdatascience.com/how-to-add-a-new-column-to-an-existing-pandas-dataframe-310a8e7baf8f)

m = pd.Series(cdat['REPORT_DAT'].str.split().apply(lambda x: x[0]))
# print(m.info())
m = m.astype('string')
mo = pd.Series(m.str.split('/').apply(lambda y: y[0]))
cdat['MONTH'] = mo.values
cdat['MONTH'] = cdat['MONTH'].astype('category')
print(cdat.head())
print(cdat.info())

print(cdat.describe(include='all'))

# visualizing the data

# separating data for reasons i.e. trying to create a scatterplot and failing
#cdat3 = cdat.copy()
#cdat31 = cdat3.values[:, :]
#cdat32 = cdat3.values[:, 2]
#cdat4 = TargetEncoder.fit_transform(cdat31, cdat32)

# looking at the distribution of the types or crime and their relative frequency
off = sns.countplot(cdat2['OFFENSE'])
off.xaxis.set_ticklabels(off.xaxis.get_ticklabels(), rotation=270, ha='right', fontsize=6)
plt.tight_layout()
plt.show()

offg = sns.countplot(cdat['offensegroup'])
offg.xaxis.set_ticklabels(offg.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
plt.tight_layout()
plt.show()

offd = sns.countplot(cdat['SHIFT'])
offd.xaxis.set_ticklabels(offd.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
plt.tight_layout()
plt.show()

offj = sns.countplot(cdat['METHOD'])
offj.xaxis.set_ticklabels(offj.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
plt.tight_layout()
plt.show()

offe = sns.countplot(cdat['MONTH'])
offe.xaxis.set_ticklabels(offe.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
plt.tight_layout()
plt.show()

offh = sns.countplot(cdat['YEAR'])
offh.xaxis.set_ticklabels(offh.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
plt.tight_layout()
plt.show()

offi = sns.countplot(cdat['WARD'])
offi.xaxis.set_ticklabels(offi.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
plt.tight_layout()
plt.show()

# using code from (https://thinkingneuron.com/how-to-visualize-the-relationship-between-two-categorical-variables-in-python/)

# Cross tabulation and bar chart between ward and offensegrouo
Crosstab1 = pd.crosstab(index=cdat['WARD'], columns=cdat['offensegroup'])
#print(Crosstab1)
Crosstab1.plot.bar()
plt.show()

# Cross tabulation and bar chart between year and offense
Crosstab2 = pd.crosstab(index=cdat['OFFENSE'], columns=cdat['YEAR'])
#print(Crosstab2)
Crosstab2.plot.bar()
plt.tight_layout()
plt.show()

# Cross tabulation and bar chart between ward and offense
Crosstab3 = pd.crosstab(index=cdat['OFFENSE'], columns=cdat['WARD'])
#print(Crosstab3)
Crosstab3.plot.bar()
plt.tight_layout()
plt.show()

# Cross tabulation and bar chart between offensegroup and offense
Crosstab4 = pd.crosstab(index=cdat['OFFENSE'], columns=cdat['offensegroup'])
#print(Crosstab4)
Crosstab4.plot.bar()
plt.tight_layout()
plt.show()

# Cross tabulation and bar chart between method and offensegroup
Crosstab5 = pd.crosstab(index=cdat['METHOD'], columns=cdat['offensegroup'])
#print(Crosstab5)
Crosstab5.plot.bar()
plt.tight_layout()
plt.show()

# Cross tabulation and bar chart between method and offense
Crosstab6 = pd.crosstab(index=cdat['OFFENSE'], columns=cdat['METHOD'])
#print(Crosstab6)
Crosstab6.plot.bar()
plt.tight_layout()
plt.show()

# Cross tabulation and bar chart between month and offensegroup
Crosstab7 = pd.crosstab(index=cdat['offensegroup'], columns=cdat['MONTH'])
#print(Crosstab7)
Crosstab7.plot.bar()
plt.tight_layout()
plt.show()


#x8 = np.arange(4)
#plt.bar(cdat['MONTH'], color = 'b', width = 0.25, height=len(cdat['MONTH']))
#plt.bar(cdat['offensegroup'], color = 'g', width = 0.25, height=len(cdat['offensegroup']))
#plt.bar(cdat['SHIFT'], color = 'r', width = 0.25, height=len(cdat['SHIFT']))
#plt.show()


#offk = sns.catplot(x="MONTH", y="OFFENSE", kind="swarm", data=cdat4)
#offk.xaxis.set_ticklabels(offk.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
#plt.show()

#offl = sns.catplot(x="MONTH", y="OFFENSE", kind="box", data=cdat4)
#plt.show()

#ffm = sns.catplot(x="MONTH", y="OFFENSE", kind="bar", data=cdat4)
#plt.show()

# label is OFFENSE whereas district, ward, month, year, shift etc. are features and conditions for the crime

X = cdat.values[:, [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23]]
Y = cdat.values[:, 2] # using the two crime categories as the target variable

# encoding the class
offense_le = LabelEncoder()
# fit and transform the class
Y = offense_le.fit_transform(Y)

# encoding the features using code from (https://towardsdatascience.com/categorical-feature-encoding-547707acf4e5)
le = TargetEncoder()
X = le.fit_transform(X,Y)

# splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# -----------------------------------------------------------------------

"""RANDOM FOREST MODEL"""
""" Accuracy = 93.979% w/ all features and 94.16% w/ K features """

""" Using random forest to predict whether a crime will be a property crime or a violent crime """

# perform training with random forest with all columns
# specify random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)

# plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, cdat.iloc[:, [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23]].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()

# select features to perform training with random forest with k columns
# select the training dataset on k-features
newX_train = X_train.iloc[:, clf.feature_importances_.argsort()[::-1][:12]]

# select the testing dataset on k-features
newX_test = X_test.iloc[:, clf.feature_importances_.argsort()[::-1][:12]]


# perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=100)

# train the model
clf_k_features.fit(newX_train, y_train)


#make predictions

# prediction on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)


# calculate metrics gini model

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test, y_pred_score[:, 1]) * 100)

print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test, y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test, y_pred_k_features_score[:, 1]) * 100)

# %%-----------------------------------------------------------------------
# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = cdat['offensegroup'].unique()


cdat_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(cdat_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=cdat_cm.columns, xticklabels=cdat_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------

"""KNN MODEL"""
""" Accuracy =  """




# -----------------------------------------------------------------------

"""NAIVE BAYES MODEL"""
""" Accuracy =  89.7% """

""" Using naive bayes to predict whether a crime will be a property crime or a violent crime """

# perform training with random forest with all columns
# specify random forest classifier
#clf = GaussianNB()

# perform training
#clf.fit(X_train, y_train)
# make predictions

# prediction on test
#y_pred = clf.predict(X_test)

#y_pred_score = clf.predict_proba(X_test)

# calculate metrics

#print("\n")

#print("Classification Report: ")
#print(classification_report(y_test,y_pred))
#print("\n")


#print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
#print("\n")

#print("ROC_AUC : ", roc_auc_score(y_test, y_pred_score[:,1]) * 100)
#print("\n")

# confusion matrix

#conf_matrix = confusion_matrix(y_test, y_pred)
#class_names = cdat['offensegroup'].unique()


#cdat_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

#plt.figure(figsize=(10,10))
#hm = sns.heatmap(cdat_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=cdat_cm.columns, xticklabels=cdat_cm.columns)
#hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
#hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=8)
#plt.ylabel('True label', fontsize=10)
#plt.xlabel('Predicted label', fontsize=10)
# Show heat map

#plt.show()