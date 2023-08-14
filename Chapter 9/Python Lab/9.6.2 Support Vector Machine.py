# Databricks notebook source
# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

np.random.seed(1)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = np.random.normal(size=(200,2))
X[:100] += 2
X[100:150] -= 2
y = np.concatenate([1*np.ones((150,)), np.zeros((50,))])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(X[:, 0], X[:, 1], c=['green' if val==1 else 'orange' for val in y], marker='o', s=1000)
plt.xlabel('X1', color='c', fontsize=20)
plt.ylabel('X2', color='c', fontsize=20)
plt.title('data with non-linear class family', color='m', fontsize=30)

# COMMAND ----------

from sklearn.svm import SVC

svmfit = SVC(C=10, kernel='rbf', gamma=1).fit(X_train, y_train)

# COMMAND ----------

def svmplot(svc, X, y, height=0.02, buffer=0.25):
    x_min, x_max = X[:, 0].min()-buffer, X[:, 0].max()+buffer
    y_min, y_max = X[:, 1].min()-buffer, X[:, 1].max()+buffer
    xx, yy = np.meshgrid(np.arange(x_min, x_max, height), np.arange(y_min, y_max, height))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
    
plt.xkcd()
plt.figure(figsize=(25, 10))
svmplot(svmfit, X, y)
plt.scatter(X[:, 0], X[:, 1], c=['green' if val==1 else 'orange' for val in y], marker='o', s=1000)
plt.xlabel('X1', color='c', fontsize=20)
plt.ylabel('X2', color='c', fontsize=20)
plt.title('support vector machine', color='m', fontsize=30)

# COMMAND ----------

svmfit.support_ # these are the support vectors

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

conf_mat = pd.DataFrame(confusion_matrix(y_test, svmfit.predict(X_test)).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat

# COMMAND ----------

class_mat = classification_report(y_test, svmfit.predict(X_test))
print(class_mat)

# COMMAND ----------

# making the SVM more flexible
svmfit = SVC(C=100, kernel='rbf', gamma=1).fit(X_train, y_train)

plt.xkcd()
plt.figure(figsize=(25, 10))
svmplot(svmfit, X, y)
plt.scatter(X[:, 0], X[:, 1], c=['green' if val==1 else 'orange' for val in y], marker='o', s=1000)
plt.xlabel('X1', color='c', fontsize=20)
plt.ylabel('X2', color='c', fontsize=20)
plt.title('support vector machine', color='m', fontsize=30)

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(y_test, svmfit.predict(X_test)).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat

# COMMAND ----------

class_mat = classification_report(y_test, svmfit.predict(X_test))
print(class_mat)

# COMMAND ----------

# making the SVM less flexible
svmfit = SVC(C=0.1, kernel='rbf', gamma=1).fit(X_train, y_train)

plt.xkcd()
plt.figure(figsize=(25, 10))
svmplot(svmfit, X, y)
plt.scatter(X[:, 0], X[:, 1], c=['green' if val==1 else 'orange' for val in y], marker='o', s=1000)
plt.xlabel('X1', color='c', fontsize=20)
plt.ylabel('X2', color='c', fontsize=20)
plt.title('support vector machine', color='m', fontsize=30)

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(y_test, svmfit.predict(X_test)).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

class_mat = classification_report(y_test, svmfit.predict(X_test))
print(class_mat)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using cross-validation to find the optimal cost

# COMMAND ----------

from sklearn.model_selection import GridSearchCV as GSV

# COMMAND ----------

cost_range = [{'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.5, 1,2,3,4]}]
cost_cv= GSV(SVC(kernel='rbf'), cost_range, cv=10, scoring='accuracy', return_train_score=True).fit(X, y)

# COMMAND ----------

best_params = cost_cv.best_params_
best_params['C']

# COMMAND ----------

# MAGIC %md
# MAGIC **GridSearchCV suggests that the best results are obtained at C=10.**

# COMMAND ----------

svmfit = SVC(C=best_params['C'], kernel='rbf', gamma=1).fit(X_train, y_train)

plt.xkcd()
plt.figure(figsize=(25, 10))
svmplot(svmfit, X, y)
plt.scatter(X[:, 0], X[:, 1], c=['green' if val==1 else 'orange' for val in y], marker='o', s=1000)
plt.xlabel('X1', color='c', fontsize=20)
plt.ylabel('X2', color='c', fontsize=20)
plt.title('support vector machine', color='m', fontsize=30)

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(y_test, svmfit.predict(X_test)).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

class_mat = classification_report(y_test, svmfit.predict(X_test))
print(class_mat)

# COMMAND ----------

# MAGIC %md
# MAGIC **Therefore, the best results are obtained at C=1.**