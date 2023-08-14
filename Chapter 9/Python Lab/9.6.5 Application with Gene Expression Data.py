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

X_train = pd.read_csv('/Users/arpanganguli/Documents/Professional/Finance/ISLR/Khanxtrain.csv', index_col=0).dropna()
X_test = pd.read_csv('/Users/arpanganguli/Documents/Professional/Finance/ISLR/Khanxtest.csv', index_col=0).dropna()

y_train = pd.read_csv('/Users/arpanganguli/Documents/Professional/Finance/ISLR/Khanytrain.csv', index_col=0).dropna()
y_test = pd.read_csv('/Users/arpanganguli/Documents/Professional/Finance/ISLR/Khanytest.csv', index_col=0).dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Support vector classification

# COMMAND ----------

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

svmfit = SVC(kernel='linear', C=10).fit(X_train, np.ravel(y_train))

# COMMAND ----------

# confusion matrix for training data
conf_mat = pd.DataFrame(confusion_matrix(y_train, svmfit.predict(X_train)).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat

# COMMAND ----------

# classification report for training data
class_mat = classification_report(y_train, svmfit.predict(X_train))
print(class_mat)

# COMMAND ----------

# MAGIC %md
# MAGIC **There are no training errors. This is because of the large number of predictors as opposed to the number of data. So, I will now fit the model on test data.**

# COMMAND ----------

# confusion matrix for test data
conf_mat = pd.DataFrame(confusion_matrix(y_test, svmfit.predict(X_test)).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat

# COMMAND ----------

# classification report for test data
class_mat = classification_report(y_test, svmfit.predict(X_test))
print(class_mat)

# COMMAND ----------

# MAGIC %md
# MAGIC **Cost=10 yields 2 test erros on this data.**