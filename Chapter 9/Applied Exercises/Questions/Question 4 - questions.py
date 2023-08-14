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

# TODO: your response here