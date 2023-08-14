# Databricks notebook source
# MAGIC %md
# MAGIC  In this problem, you will use support vector approaches in order to
# MAGIC predict whether a given car gets high or low gas mileage based on the
# MAGIC `Auto` data set.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["name"]
num_cols = list(set(Auto.columns) - set(str_cols))
Auto[str_cols] = Auto[str_cols].astype(str)
Auto[num_cols] = Auto[num_cols].astype(float)

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a binary variable that takes on a 1 for cars with gas
# MAGIC mileage above the median, and a 0 for cars with gas mileage
# MAGIC below the median.**

# COMMAND ----------

mpg_median = Auto.mpg.median()
mpg_median

# COMMAND ----------

Auto['med'] = np.where(Auto.mpg > mpg_median, 1, 0)
Auto.head()

# COMMAND ----------

Auto.med.value_counts()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(Auto.mpg, Auto.med, color='green', s=250, alpha=0.5)
plt.vlines(x=mpg_median, ymin=0.0, ymax=1.0, colors='orange', linestyles='dotted', label='median')
plt.legend()
plt.xlabel('miles per gallon', fontsize=20, color='c')
plt.ylabel('greater than or lesser than median', fontsize=20, color='c')
plt.title('miles per gallon and median value', fontsize=30, color='m')

# COMMAND ----------

Auto.horsepower.dtype

# COMMAND ----------

Auto['hp'] = Auto.horsepower.astype(float)

# COMMAND ----------

Auto.head()

# COMMAND ----------

Auto.hp.dtype

# COMMAND ----------

Auto.drop(columns='horsepower', inplace=True)
Auto.head()

# COMMAND ----------

Auto.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a support vector classifer to the data with various values of
# MAGIC C, in order to predict whether a car gets high or low gas mileage.
# MAGIC Report the cross-validation errors associated with diferent values of this parameter. Comment on your results. Note you will
# MAGIC need to ft the classifer without the gas mileage variable to produce sensible results.**

# COMMAND ----------

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV as GSV
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

df = Auto.drop(columns=['name', 'mpg', 'med'])
Y = Auto['med']

# COMMAND ----------

# cost = 10
svmfit10 = SVC(C=10, kernel='linear').fit(df, Y)

# COMMAND ----------

conf_mat10 = pd.DataFrame(confusion_matrix(Y, svmfit10.predict(df)).T, index = svmfit10.classes_, columns = svmfit10.classes_)
conf_mat10

# COMMAND ----------

class_mat10 = classification_report(Y, svmfit10.predict(df))
print(class_mat10)

# COMMAND ----------

# cost = 1
svmfit1 = SVC(C=1, kernel='linear').fit(df, Y)

# COMMAND ----------

conf_mat1 = pd.DataFrame(confusion_matrix(Y, svmfit1.predict(df)).T, index = svmfit1.classes_, columns = svmfit1.classes_)
conf_mat1

# COMMAND ----------

class_mat1 = classification_report(Y, svmfit1.predict(df))
print(class_mat1)

# COMMAND ----------

# cost = 0.25
svmfit025 = SVC(C=0.25, kernel='linear').fit(df, Y)

# COMMAND ----------

conf_mat025 = pd.DataFrame(confusion_matrix(Y, svmfit025.predict(df)).T, index = svmfit025.classes_, columns = svmfit025.classes_)
conf_mat025

# COMMAND ----------

class_mat025 = classification_report(Y, svmfit025.predict(df))
print(class_mat025)

# COMMAND ----------

# cost = 20
svmfit20 = SVC(C=20, kernel='linear').fit(df, Y)

# COMMAND ----------

conf_mat20 = pd.DataFrame(confusion_matrix(Y, svmfit20.predict(df)).T, index = svmfit20.classes_, columns = svmfit20.classes_)
conf_mat20

# COMMAND ----------

class_mat20 = classification_report(Y, svmfit20.predict(df))
print(class_mat20)

# COMMAND ----------

# cost = 1000
svmfit1000 = SVC(C=1000, kernel='linear').fit(df, Y)

# COMMAND ----------

conf_mat1000 = pd.DataFrame(confusion_matrix(Y, svmfit1000.predict(df)).T, index = svmfit1000.classes_, columns = svmfit1000.classes_)
conf_mat1000

# COMMAND ----------

class_mat1000 = classification_report(Y, svmfit1000.predict(df))
print(class_mat1000)

# COMMAND ----------

cost_range = [{'C': [0.01, 0.1, 1, 5, 10, 100, 1000, 10000]}]
cost_cv= GSV(SVC(kernel='linear'), cost_range, cv=10, scoring='accuracy').fit(df, Y)

# COMMAND ----------

means = pd.DataFrame([cost_cv.cv_results_['mean_test_score']]).T
means.columns = ['means']
means.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(means, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('mean test score', fontsize=30, color='m')
plt.xlabel('cost index (for actual costs, see code)', fontsize=20, color='c')
plt.ylabel('accuracy score', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC *I get the highest accuracy at $cost$=1.*

# COMMAND ----------

# MAGIC %md
# MAGIC **c.  Now repeat (b), this time using SVMs with radial and polynomial basis kernels, with diferent values of gamma and degree and
# MAGIC C. Comment on your results.**

# COMMAND ----------

cost_range = [{'C': [0.01, 0.1, 1, 5, 10, 100, 1000, 10000], 'gamma': [0.05, 0.75, 1, 3, 5, 10, 12, 14], 
              'degree': [3, 5, 7, 10, 12, 15, 17, 25]}]
cost_cv= GSV(SVC(kernel='rbf'), cost_range, cv=10, scoring='accuracy').fit(df, Y)

# COMMAND ----------

means = pd.DataFrame([cost_cv.cv_results_['mean_test_score']]).T
means.columns = ['means']
means.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(means, ls='-.', color='green', marker='o', markersize=10, markerfacecolor='orange')
plt.title('mean test score', fontsize=30, color='m')
plt.xlabel('cost index (for actual costs, see code)', fontsize=20, color='c')
plt.ylabel('accuracy score', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Make some plots to back up your assertions in (b) and (c).**

# COMMAND ----------

Ypred = svmfit10.predict(df)

# COMMAND ----------

list(df)

# COMMAND ----------

xx = df[['cylinders', 'displacement']]

# COMMAND ----------

svmfitxx = SVC(kernel='linear').fit(xx, Y)

# COMMAND ----------

svmpredxx = svmfitxx.predict(xx)

# COMMAND ----------

dfpred = pd.concat([xx, pd.DataFrame([svmpredxx]).T], axis=1)

# COMMAND ----------

dfpred.columns = ['cylinders', 'displacement', 'predict']
dfpred.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(dfpred.cylinders[dfpred.predict==0], dfpred.displacement[dfpred.predict==0], color='green', s=200, alpha=0.5)
plt.scatter(dfpred.cylinders[dfpred.predict==1], dfpred.displacement[dfpred.predict==1], color='orange', s=200, alpha=0.5)

# COMMAND ----------

list(df)

# COMMAND ----------

xx = df[['weight', 'displacement']]

# COMMAND ----------

svmfitxx = SVC(kernel='linear').fit(xx, Y)

# COMMAND ----------

svmpredxx

# COMMAND ----------

dfpred = pd.concat([xx, pd.DataFrame([svmpredxx]).T], axis=1)

# COMMAND ----------

dfpred.columns = ['weight', 'displacement', 'predict']
dfpred.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(dfpred.weight[dfpred.predict==0], dfpred.displacement[dfpred.predict==0], color='green', s=200, alpha=0.5)
plt.scatter(dfpred.weight[dfpred.predict==1], dfpred.displacement[dfpred.predict==1], color='orange', s=200, alpha=0.5)

# COMMAND ----------

list(df)

# COMMAND ----------

xx = df[['acceleration', 'hp']]

# COMMAND ----------

svmfitxx = SVC(kernel='linear').fit(xx, Y)

# COMMAND ----------

svmpredxx

# COMMAND ----------

dfpred = pd.concat([xx, pd.DataFrame([svmpredxx]).T], axis=1)

# COMMAND ----------

dfpred.columns = ['acceleration', 'hp', 'predict']
dfpred.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(dfpred.acceleration[dfpred.predict==0], dfpred.hp[dfpred.predict==0], color='green', s=200, alpha=0.5)
plt.scatter(dfpred.acceleration[dfpred.predict==1], dfpred.hp[dfpred.predict==1], color='orange', s=200, alpha=0.5)

# COMMAND ----------

list(df)

# COMMAND ----------

xx = df[['acceleration', 'hp']]

# COMMAND ----------

svmfitxx = SVC(kernel='rbf', C=1000, degree=3, gamma=10).fit(xx, Y)

# COMMAND ----------

svmpredxx

# COMMAND ----------

dfpred = pd.concat([xx, pd.DataFrame([svmpredxx]).T], axis=1)

# COMMAND ----------

dfpred.columns = ['acceleration', 'hp', 'predict']
dfpred.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(dfpred.acceleration[dfpred.predict==0], dfpred.hp[dfpred.predict==0], color='green', s=200, alpha=0.5)
plt.scatter(dfpred.acceleration[dfpred.predict==1], dfpred.hp[dfpred.predict==1], color='orange', s=200, alpha=0.5)