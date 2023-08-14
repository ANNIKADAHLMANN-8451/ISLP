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

X = np.random.normal(size=(20,2))
X

# COMMAND ----------

y_1 = np.array([1]*10)
y_minus1 = np.array([-1]*10)
y = np.concatenate([y_1, y_minus1])
y

# COMMAND ----------

X = np.concatenate([X, np.random.normal(loc=(0, 2), size=(50,2))])
y = np.concatenate([y, 2*np.ones((50,))])

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(X[:, 0], X[:, 1], s=1000, c=['green' if val==1 else ('orange' if val==0 else 'yellow') for val in y])
plt.title('data with multiple classes', fontsize=30, color='m')
plt.xlabel('X', fontsize=20, color='c')
plt.ylabel('y', fontsize=20, color='c')

# COMMAND ----------

from sklearn.svm import SVC
svmfit = SVC(C=1, kernel='rbf', gamma='auto').fit(X, y)

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
plt.scatter(X[:, 0], X[:, 1], s=1000, c=['green' if val==1 else ('orange' if val==0 else 'yellow') for val in y])
plt.title('data with multiple classes', fontsize=30, color='m')
plt.xlabel('X', fontsize=20, color='c')
plt.ylabel('y', fontsize=20, color='c')

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(y, svmfit.predict(X)).T, index = svmfit.classes_, columns = svmfit.classes_)
conf_mat

# COMMAND ----------

class_mat = classification_report(y, svmfit.predict(X))
print(class_mat)

# COMMAND ----------

