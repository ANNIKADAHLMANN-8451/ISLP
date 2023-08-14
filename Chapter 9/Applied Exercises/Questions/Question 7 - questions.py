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

# TODO: your response here

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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c.  Now repeat (b), this time using SVMs with radial and polynomial basis kernels, with diferent values of gamma and degree and
# MAGIC C. Comment on your results.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Make some plots to back up your assertions in (b) and (c).**

# COMMAND ----------

# TODO: your response here