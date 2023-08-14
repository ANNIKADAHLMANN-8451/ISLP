# Databricks notebook source
# MAGIC %md
# MAGIC This problem involves the OJ data set.

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

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/OJ.csv"
OJ = spark.read.option("header", "true").csv(url).toPandas()
OJ.set_index("SlNo", inplace=True)

str_cols = ["Purchase", "Store7"]
float_cols = ["PriceCH", "PriceMM", "DiscCH", "DiscMM", "LoyalCH", "SalePriceMM", "SalePriceCH", "PriceDiff", "PctDiscMM", "PctDiscCH", "ListPriceDiff"]
int_cols = list(set(OJ.columns)-set(str_cols)-set(float_cols))
OJ[str_cols] = OJ[str_cols].astype(str)
OJ[float_cols] = OJ[float_cols].astype(float)
OJ[int_cols] = OJ[int_cols].astype(int)

# COMMAND ----------

OJ.head()

# COMMAND ----------

OJ.Purchase = pd.factorize(OJ.Purchase)[0]
OJ.Store7 = pd.factorize(OJ.Store7)[0]
OJ.head()

# COMMAND ----------

OJ.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a training set containing a random sample of 800
# MAGIC observations, and a test set containing the remaining
# MAGIC observations.**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X = OJ.drop(columns='Purchase')
y = OJ.Purchase

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.53271028037, random_state=42)

# COMMAND ----------

X_train.shape

# COMMAND ----------

y_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a support vector classifer to the training data using
# MAGIC `C = 0.01`, with `Purchase` as the response and the other variables
# MAGIC as predictors. How many support points are there?**

# COMMAND ----------

from sklearn.svm import SVC

# COMMAND ----------

svmfit = SVC(kernel='linear', C=0.01).fit(X_train, y_train)

# COMMAND ----------

svmfit.support_vectors_

# COMMAND ----------

svmfit.classes_

# COMMAND ----------

svmfit.coef_

# COMMAND ----------

svmfit.get_params()

# COMMAND ----------

# MAGIC %md
# MAGIC **c. What are the training and test error rates?**

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

svmpred_train = svmfit.predict(X_train)

# COMMAND ----------

conf_mat_train = pd.DataFrame(confusion_matrix(y_train, svmpred_train).T, index=svmfit.classes_, columns=svmfit.classes_)
conf_mat_train

# COMMAND ----------

class_mat_train  = classification_report(y_train, svmpred_train)
print(class_mat_train)

# COMMAND ----------

svmpred_test = svmfit.predict(X_test)

# COMMAND ----------

conf_mat_test = pd.DataFrame(confusion_matrix(y_test, svmpred_test).T, index=svmfit.classes_, columns=svmfit.classes_)
conf_mat_test

# COMMAND ----------

class_mat_test  = classification_report(y_test, svmpred_test)
print(class_mat_test)

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Use cross-validation to select an optimal C. Consider values in
# MAGIC the range 0.01 to 10.**

# COMMAND ----------

cost_range = np.linspace(0.01, 10, 50)

# COMMAND ----------

errdf = pd.DataFrame()
for k in cost_range:
    svmfit = SVC(kernel='linear', C=k, degree=1).fit(X_train, y_train)
    svmpred_train = svmfit.predict(X_train)
    conf_mat_train = confusion_matrix(y_train, svmpred_train)
    NoNo = pd.DataFrame([conf_mat_train[0][0]])
    YesYes = pd.DataFrame([conf_mat_train[1][1]])
    errdf = errdf.append((NoNo + YesYes) / 500)

errdf.columns = ['error']
errdf.reset_index(drop=True, inplace=True)
errdf.index = np.round(cost_range, 2)
errdf

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(errdf.index, errdf.error, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('error rates for svc prediction on training data set', fontsize=30, color='m')
plt.xlabel('cost index (for actual costs, see code)', fontsize=20, color='c')
plt.ylabel('error', fontsize=20, color='c')

# COMMAND ----------

errdf = pd.DataFrame()
for k in cost_range:
    svmfit = SVC(kernel='linear', C=k, degree=1).fit(X_train, y_train)
    svmpred_test = svmfit.predict(X_test)
    conf_mat_test = confusion_matrix(y_test, svmpred_test)
    NoNo = pd.DataFrame([conf_mat_test[0][0]])
    YesYes = pd.DataFrame([conf_mat_test[1][1]])
    errdf = errdf.append((NoNo + YesYes) / 500)

errdf.columns = ['error']
errdf.reset_index(drop=True, inplace=True)
errdf.index = np.round(cost_range, 2)
errdf

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(errdf.index, errdf.error, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('error rates for svc prediction on test data set', fontsize=30, color='m')
plt.xlabel('cost index (for actual costs, see code)', fontsize=20, color='c')
plt.ylabel('error', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC Both training and test data sets provide least error for $cost$=0.01.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Compute the training and test error rates using this new value
# MAGIC for C.**

# COMMAND ----------

svmpred001_train = svmfit.predict(X_train)
svmpred001_test = svmfit.predict(X_test)

# COMMAND ----------

conf_mat_train = pd.DataFrame(confusion_matrix(y_train, svmpred_train), index=svmfit.classes_, columns=svmfit.classes_)
conf_mat_test = pd.DataFrame(confusion_matrix(y_test, svmpred_test), index=svmfit.classes_, columns=svmfit.classes_)

# COMMAND ----------

conf_mat_train

# COMMAND ----------

class_mat_train = classification_report(y_train, svmpred_train)
print(class_mat_train)

# COMMAND ----------

conf_mat_test

# COMMAND ----------

class_mat_test = classification_report(y_test, svmpred_test)
print(class_mat_test)

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Repeat parts (b) through (e) using a support vector machine
# MAGIC with a radial kernel. Use the default value for gamma.**

# COMMAND ----------

svmfit = SVC(kernel='rbf', C=0.01, degree=3, gamma=0.01).fit(X_train, y_train)

# COMMAND ----------

svmfit.support_vectors_

# COMMAND ----------

svmfit.classes_

# COMMAND ----------

svmfit.get_params()

# COMMAND ----------

svmpred_train = svmfit.predict(X_train)

# COMMAND ----------

conf_mat_train = pd.DataFrame(confusion_matrix(y_train, svmpred_train).T, index=svmfit.classes_, columns=svmfit.classes_)
conf_mat_train

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

class_mat_train  = classification_report(y_train, svmpred_train)
print(class_mat_train)

# COMMAND ----------

svmpred_test = svmfit.predict(X_test)

# COMMAND ----------

conf_mat_test = pd.DataFrame(confusion_matrix(y_test, svmpred_test).T, index=svmfit.classes_, columns=svmfit.classes_)
conf_mat_test

# COMMAND ----------

class_mat_test  = classification_report(y_test, svmpred_test)
print(class_mat_test)

# COMMAND ----------

cost_range = np.linspace(0.01, 10, 50)
gamma = np.linspace(0.1, 0.5, 5)

# COMMAND ----------

errdf = pd.DataFrame()
for k in cost_range:
    for j in gamma:
        svmfit = SVC(kernel='rbf', C=k, degree=3).fit(X_train, y_train)
        svmpred_train = svmfit.predict(X_train)
        conf_mat_train = confusion_matrix(y_train, svmpred_train)
        NoNo = pd.DataFrame([conf_mat_train[0][0]])
        YesYes = pd.DataFrame([conf_mat_train[1][1]])
        errdf = errdf.append((NoNo + YesYes) / 500)

errdf.columns = ['error']
errdf.reset_index(drop=True, inplace=True)
errdf

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(errdf.index, errdf.error, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('error rates for svc prediction on training data set', fontsize=30, color='m')
plt.xlabel('cost index (for actual costs, see code)', fontsize=20, color='c')
plt.ylabel('error', fontsize=20, color='c')

# COMMAND ----------

errdf = pd.DataFrame()
for k in cost_range:
    svmfit = SVC(kernel='rbf', C=k, degree=3).fit(X_train, y_train)
    svmpred_test = svmfit.predict(X_test)
    conf_mat_test = confusion_matrix(y_test, svmpred_test)
    NoNo = pd.DataFrame([conf_mat_test[0][0]])
    YesYes = pd.DataFrame([conf_mat_test[1][1]])
    errdf = errdf.append((NoNo + YesYes) / 500)

errdf.columns = ['error']
errdf.reset_index(drop=True, inplace=True)
errdf.index = np.round(cost_range, 2)
errdf

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(errdf.index, errdf.error, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('error rates for svc prediction on test data set', fontsize=30, color='m')
plt.xlabel('cost index (for actual costs, see code)', fontsize=20, color='c')
plt.ylabel('error', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC Both training and test data sets provide least error for $cost$=0.01.

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Repeat parts (b) through (e) using a support vector machine
# MAGIC with a polynomial kernel. Set degree = 2.**

# COMMAND ----------

svmfit = SVC(kernel='rbf', C=0.01, degree=2, gamma=0.01).fit(X_train, y_train)

# COMMAND ----------

svmfit.support_vectors_

# COMMAND ----------

svmfit.classes_

# COMMAND ----------

svmfit.get_params()

# COMMAND ----------

svmpred_train = svmfit.predict(X_train)

# COMMAND ----------

conf_mat_train = pd.DataFrame(confusion_matrix(y_train, svmpred_train).T, index=svmfit.classes_, columns=svmfit.classes_)
conf_mat_train

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

class_mat_train  = classification_report(y_train, svmpred_train)
print(class_mat_train)

# COMMAND ----------

svmpred_test = svmfit.predict(X_test)

# COMMAND ----------

conf_mat_test = pd.DataFrame(confusion_matrix(y_test, svmpred_test).T, index=svmfit.classes_, columns=svmfit.classes_)
conf_mat_test

# COMMAND ----------

class_mat_test  = classification_report(y_test, svmpred_test)
print(class_mat_test)

# COMMAND ----------

cost_range = np.linspace(0.01, 10, 50)
gamma = np.linspace(0.1, 0.5, 5)

# COMMAND ----------

errdf = pd.DataFrame()
for k in cost_range:
    for j in gamma:
        svmfit = SVC(kernel='rbf', C=k, degree=2).fit(X_train, y_train)
        svmpred_train = svmfit.predict(X_train)
        conf_mat_train = confusion_matrix(y_train, svmpred_train)
        NoNo = pd.DataFrame([conf_mat_train[0][0]])
        YesYes = pd.DataFrame([conf_mat_train[1][1]])
        errdf = errdf.append((NoNo + YesYes) / 500)

errdf.columns = ['error']
errdf.reset_index(drop=True, inplace=True)
errdf

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(errdf.index, errdf.error, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('error rates for svc prediction on training data set', fontsize=30, color='m')
plt.xlabel('cost index (for actual costs, see code)', fontsize=20, color='c')
plt.ylabel('error', fontsize=20, color='c')

# COMMAND ----------

errdf = pd.DataFrame()
for k in cost_range:
    svmfit = SVC(kernel='rbf', C=k, degree=2).fit(X_train, y_train)
    svmpred_test = svmfit.predict(X_test)
    conf_mat_test = confusion_matrix(y_test, svmpred_test)
    NoNo = pd.DataFrame([conf_mat_test[0][0]])
    YesYes = pd.DataFrame([conf_mat_test[1][1]])
    errdf = errdf.append((NoNo + YesYes) / 500)

errdf.columns = ['error']
errdf.reset_index(drop=True, inplace=True)
errdf.index = np.round(cost_range, 2)
errdf

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(errdf.index, errdf.error, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('error rates for svc prediction on test data set', fontsize=30, color='m')
plt.xlabel('cost index (for actual costs, see code)', fontsize=20, color='c')
plt.ylabel('error', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC Both training and test data sets provide least error for $cost$=0.01.

# COMMAND ----------

# MAGIC %md
# MAGIC **h.  Overall, which approach seems to give the best results on this
# MAGIC data?**

# COMMAND ----------

# MAGIC %md
# MAGIC Overall, radial kernel seems to provide least error on both training and test data set.