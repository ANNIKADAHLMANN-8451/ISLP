# Databricks notebook source
# MAGIC %md
# MAGIC ### Preprocessing

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

from sklearn.model_selection import train_test_split

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
df = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["name"]
num_cols = list(set(df.columns) - set(str_cols))
df[str_cols] = df[str_cols].astype(str)
df[num_cols] = df[num_cols].astype(float)

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using simple linear regression

# COMMAND ----------

X = df[['horsepower']]
y = df['mpg']

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

lmfit = LinearRegression().fit(X_train, y_train)

# COMMAND ----------

lmpred = lmfit.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_squared_error

# COMMAND ----------

MSE = mean_squared_error(y_test, lmpred)

# COMMAND ----------

round(MSE, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Leave-One-Out-Cross-Validation (LOOCV)

# COMMAND ----------

from sklearn.model_selection import LeaveOneOut

# COMMAND ----------

X = df[['horsepower']]
y = df['mpg']

# COMMAND ----------

loo = LeaveOneOut()

# COMMAND ----------

total_sets = loo.get_n_splits(X)

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

from sklearn.metrics import mean_squared_error

# COMMAND ----------

MSE = 0

# COMMAND ----------

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lmfit = LinearRegression().fit(X_train, y_train)
    lmpred = lmfit.predict(X_test)
    MSE += mean_squared_error(y_test, lmpred)

# COMMAND ----------

MSE

# COMMAND ----------

MSE_mean = MSE/total_sets

# COMMAND ----------

round(MSE_mean, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC **As we can see, LOOCV provides similar MSE to simple linear regression. Therefore, in the absence of a test dataset, we can resample the existing dataset through LOOCV to generate similar results to simple linear regression.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Leave-One-Out-Cross-Validation (LOOCV) for polynomial regressions (order: 1-5)

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

X = df[['horsepower']]
y = df['mpg']

# COMMAND ----------

loo = LeaveOneOut()

# COMMAND ----------

total_sets = loo.get_n_splits(X)

# COMMAND ----------

MSE_all = pd.DataFrame()

# COMMAND ----------

for i in range(1,6):
    MSE = 0
    X = df[['horsepower']]
    X_ = pd.DataFrame(PF(i).fit_transform(X))
    X_.drop(columns=0, inplace=True)
    y = df[['mpg']]
    for train_index, test_index in loo.split(X):
        X_train, X_test = X_.iloc[train_index], X_.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lmfit = LinearRegression().fit(X_train, y_train)
        lmpred = lmfit.predict(X_test)
        MSE += mean_squared_error(y_test, lmpred)
    MSE_mean = MSE/total_sets
    MSE_all = MSE_all.append([MSE_mean])

# COMMAND ----------

MSE_all.columns = [['MSE']]
MSE_all.reset_index(drop=True, inplace=True)
round(MSE_all, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC **As we can see, there is a sharp drop between linear and quadratic regressions. Rest of the higher-order regressions
# MAGIC have similar MSEs.**