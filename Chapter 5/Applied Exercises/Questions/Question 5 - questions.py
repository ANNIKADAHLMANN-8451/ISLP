# Databricks notebook source
# MAGIC %md
# MAGIC In Chapter 4, we used logistic regression to predict the probability of
# MAGIC `default` using `income` and `balance` on the `Default` data set. We will
# MAGIC now estimate the test error of this logistic regression model using the
# MAGIC validation set approach. Do not forget to set a random seed before
# MAGIC beginning your analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd

# COMMAND ----------

# import relevant data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Default.csv"
Default = spark.read.option("header", "true").csv(url).toPandas()
Default.set_index('_c0', inplace=True)

float_cols = ["balance", "income"]
str_cols = ["default", "student"]
Default[float_cols] = Default[float_cols].astype(float)
Default[str_cols] = Default[str_cols].astype(str)

# COMMAND ----------

Default.head()

# COMMAND ----------

Default.info()

# COMMAND ----------

dfX = Default[['student', 'balance','income']]
dfX = pd.get_dummies(data = dfX, drop_first=True)
dfy = Default['default']

# COMMAND ----------

dfX.head()

# COMMAND ----------

dfy.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Fit a logistic regression model that uses `income` and `balance` to
# MAGIC predict `default`.**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Using the validation set approach, estimate the test error of this
# MAGIC model. In order to do this, you must perform the following steps:**
# MAGIC - i. Split the sample set into a training set and a validation set.
# MAGIC - ii. Fit a multiple logistic regression model using only the training observations.
# MAGIC - iii. Obtain a prediction of `default` status for each individual in the validation set by computing the posterior probability of default for that individual, and classifying the individual to the default category if the posterior probability is greater than 0.5.
# MAGIC - iv. Compute the validation set error, which is the fraction of the observations in the validation set that are misclassifed.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Repeat the process in (b) three times, using three diferent splits
# MAGIC of the observations into a training set and a validation set. Comment on the results obtained.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Now consider a logistic regression model that predicts the probability of `default` using `income`, `balance`, and a dummy variable
# MAGIC for `student`. Estimate the test error for this model using the validation set approach. Comment on whether or not including a
# MAGIC dummy variable for `student` leads to a reduction in the test error
# MAGIC rate.**

# COMMAND ----------

# TODO: your response here