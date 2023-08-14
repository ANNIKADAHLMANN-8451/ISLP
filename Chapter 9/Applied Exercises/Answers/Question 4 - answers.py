# Databricks notebook source
# MAGIC %md
# MAGIC Generate a simulated two-class data set with 100 observations and
# MAGIC two features in which there is a visible but non-linear separation between the two classes. Show that in this setting, a support vector
# MAGIC machine with a polynomial kernel (with degree greater than 1) or a
# MAGIC radial kernel will outperform a support vector classifer on the training data. Which technique performs best on the test data? Make
# MAGIC plots and report training and test error rates in order to back up
# MAGIC your assertions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# COMMAND ----------

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

X = np.random.normal(size=100)
X1 = X[0:50]
X2 = X[51:100]

y = 2 * pow(X,2) + 3.5 + np.random.normal(size=100)
y1 = y[0:50]
y2 = y[51:100]


y1 += 3.7
y2 -= 3.7

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(X1, y1, cmap=mpl.cm.Paired, marker='o', s=500)
plt.scatter(X2, y2, cmap=mpl.cm.Paired, marker='o', s=500)
plt.xlabel('X', color='green', fontsize=20)
plt.ylabel('y', color='orange', fontsize=20)
plt.title('data with visible but non-linear separation', color='m', fontsize=30)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Support vector machine with a non-linear kernel

# COMMAND ----------

Z = y = np.concatenate([1*np.ones((50,)), -1*np.zeros((50,))])
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.5, random_state=42)
svmfit = SVC(C=40, kernel='rbf', gamma=1).fit(X_train.reshape(-1, 1), Z_train)

# COMMAND ----------

svmfit.support_

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

conf_mat_train = pd.DataFrame(confusion_matrix(Z_train, svmfit.predict(X_train.reshape(-1,1))).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat_train

# COMMAND ----------

class_mat_train = classification_report(Z_train, svmfit.predict(X_train.reshape(-1, 1)))
print(class_mat_train)

# COMMAND ----------

conf_mat_test = pd.DataFrame(confusion_matrix(Z_test, svmfit.predict(X_test.reshape(-1,1))).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat_test

# COMMAND ----------

class_mat_test = classification_report(Z_test, svmfit.predict(X_test.reshape(-1, 1)))
print(class_mat_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Support vector classifier (linear kernel)

# COMMAND ----------

svmfit_linear = SVC(kernel='linear', C=40).fit(X_train.reshape(-1, 1), Z_train)

# COMMAND ----------

conf_mat_linear_train = pd.DataFrame(confusion_matrix(Z_train, svmfit_linear.predict(X_train.reshape(-1,1))).T, index = svmfit_linear.classes_, columns = svmfit.classes_)
conf_mat_linear_train

# COMMAND ----------

class_mat_linear_train = classification_report(Z_train, svmfit_linear.predict(X_train.reshape(-1, 1)))
print(class_mat_linear_train)

# COMMAND ----------

conf_mat_linear_test = pd.DataFrame(confusion_matrix(Z_test, svmfit_linear.predict(X_test.reshape(-1,1))).T, index = svmfit_linear.classes_, columns = svmfit.classes_)
conf_mat_linear_test

# COMMAND ----------

class_mat_linear_test = classification_report(Z_test, svmfit_linear.predict(X_test.reshape(-1, 1)))
print(class_mat_linear_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, there is no difference between the performance of a linear and non-linear kernel on training data. But, non-linear kernel outperforms linear kernel on test data.

# COMMAND ----------

