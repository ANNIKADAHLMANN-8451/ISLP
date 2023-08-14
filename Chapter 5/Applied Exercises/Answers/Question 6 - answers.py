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

X = dfX[['balance', 'income']]
X = sm.add_constant(X)
y = pd.get_dummies(dfy, drop_first=True)

# COMMAND ----------

glmfit = sm.GLM(y, X, family=sm.families.Binomial()).fit()

# COMMAND ----------

glmfit.summary()

# COMMAND ----------

estimated_std_err = np.array(glmfit.params / glmfit.tvalues)

# COMMAND ----------

estimated_std_err

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Write a function, `boot_fn()`, that takes as input the `Default` data
# MAGIC set as well as an index of the observations, and that outputs
# MAGIC the coefcient estimates for `income` and `balance` in the multiple
# MAGIC logistic regression model.**

# COMMAND ----------

def bootfn(data, index):
    X = data[['balance', 'income']]
    X = sm.add_constant(X)
    y = pd.get_dummies(data['default'], drop_first=True)
    X_train = X.iloc[index]
    y_train = y.iloc[index]
    glmfit = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
    estimated_std_err = np.array(glmfit.params / glmfit.tvalues)
    return estimated_std_err

# COMMAND ----------

bootfn(Default, list(range(1,10000)))

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Following the bootstrap example in the lab, use your `boot_fn()`
# MAGIC function to estimate the standard errors of the logistic regression
# MAGIC coefcients for `income` and `balance`.**

# COMMAND ----------

from sklearn.utils import resample

# COMMAND ----------

std_err_df = pd.DataFrame()
idx = list(range(10000))

# COMMAND ----------

for i in range(1000):
    std_temp = bootfn(Default, resample(idx, replace=True))
    std_err_df = std_err_df.append([std_temp])

# COMMAND ----------

std_err_df.reset_index(drop=True, inplace=True)
std_err_df.columns = ['intercept', 'balance', 'income']

# COMMAND ----------

std_err_df.head()

# COMMAND ----------

std_err_df.shape

# COMMAND ----------

std_err_df.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Comment on the estimated standard errors obtained using the
# MAGIC `sm.GLM()` function and using the bootstrap.**

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, the bootstrap of the standard error estimates are close to standard errors of logistic regression coefficients.