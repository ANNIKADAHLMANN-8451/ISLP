# Databricks notebook source
# MAGIC %md
# MAGIC We continue to consider the use of a logistic regression model to
# MAGIC predict the probability of `default` using `income` and `balance` on the
# MAGIC `Default` data set. In particular, we will now compute estimates for the
# MAGIC standard errors of the `income` and `balance` logistic regression coefcients in two diferent ways: (1) using the bootstrap, and (2) using the
# MAGIC standard formula for computing the standard errors in the `sm.GLM()`
# MAGIC function. Do not forget to set a random seed before beginning your
# MAGIC analysis.

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
# MAGIC **a. Using the `.summary()` and `sm.GLM()` functions, determine the
# MAGIC estimated standard errors for the coefcients associated with
# MAGIC income and balance in a multiple logistic regression model that
# MAGIC uses both predictors.**

# COMMAND ----------

import statsmodels.api as sm

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Write a function, `boot_fn()`, that takes as input the `Default` data
# MAGIC set as well as an index of the observations, and that outputs
# MAGIC the coefcient estimates for `income` and `balance` in the multiple
# MAGIC logistic regression model.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Following the bootstrap example in the lab, use your `boot_fn()`
# MAGIC function to estimate the standard errors of the logistic regression
# MAGIC coefcients for `income` and `balance`.**

# COMMAND ----------

from sklearn.utils import resample

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Comment on the estimated standard errors obtained using the
# MAGIC `sm.GLM()` function and using the bootstrap.**

# COMMAND ----------

# TODO: your response here