# Databricks notebook source
# MAGIC %md
# MAGIC We have seen that we can ft an SVM with a non-linear kernel in order
# MAGIC to perform classifcation using a non-linear decision boundary. We will
# MAGIC now see that we can also obtain a non-linear decision boundary by
# MAGIC performing logistic regression using non-linear transformations of the
# MAGIC features.

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

# MAGIC %md
# MAGIC **a. Generate a data set with n = 500 and p = 2, such that the observations belong to two classes with a quadratic decision boundary
# MAGIC between them. For instance, you can do this as follows:**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Plot the observations, colored according to their class labels.
# MAGIC Your plot should display X1 on the x-axis, and X2 on the y-axis.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Fit a logistic regression model to the data, using X1 and X2 as
# MAGIC predictors.**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Apply this model to the training data in order to obtain a predicted class label for each training observation. Plot the observations, colored according to the predicted class labels. The
# MAGIC decision boundary should be linear.**

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now fit a logistic regression model to the data using non-linear
# MAGIC functions of X1 and X2 as predictors (e.g. X_1^2, X1×X2, log(X2),
# MAGIC and so forth).**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Fit a support vector classifer to the data with X1 and X2 as
# MAGIC predictors. Obtain a class prediction for each training observation. Plot the observations, colored according to the predicted
# MAGIC class labels.**

# COMMAND ----------

from sklearn.svm import SVC

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Fit a SVM using a non-linear kernel to the data. Obtain a class
# MAGIC prediction for each training observation. Plot the observations,
# MAGIC colored according to the predicted class labels.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **i. Comment on results.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

