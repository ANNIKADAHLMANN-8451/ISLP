# Databricks notebook source
# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import statistical packages
import numpy as np
import pandas as pd

# COMMAND ----------

# import data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC *I do not need to specify a separate 50% training dataset. Instead we use the [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) method from sklearn.*

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
# MAGIC ### Regressions using random state = 1

# COMMAND ----------

# MAGIC %md
# MAGIC **Simple Linear Regression**

# COMMAND ----------

X = df[['horsepower']]
y = df['mpg']

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# COMMAND ----------

X_train.shape

# COMMAND ----------

y_train.shape

# COMMAND ----------

X_test.shape

# COMMAND ----------

y_test.shape

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC *The Auto dataset contains 397 rows whereas the same dataset in the book example contains 392 rows. This can be explained
# MAGIC by the fact that some of the rows have missing values and have been deleted. I have, however, imputed those values. So,
# MAGIC I have the same number of rows as the original dataset. More information about imputation of missing values can be found 
# MAGIC [here](http://www.stat.columbia.edu/~gelman/arm/missing.pdf). In any case, it does not matter since the prime purpose of the chapter is to show relative differences in prediction abilities of different methodologies. So as long as the relative difference is more or less the same, the point still stands.*

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
# MAGIC **Polynomial Regression (horsepower$^2$)**

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

X = df[['horsepower']]
X_ = pd.DataFrame(PF(2).fit_transform(X))
y = df[['mpg']]

# COMMAND ----------

X_.head()

# COMMAND ----------

X_.drop(columns=0, inplace=True)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.5, random_state=1)

# COMMAND ----------

lmfit2 = LinearRegression().fit(X_train, y_train)

# COMMAND ----------

lmpred2 = lmfit2.predict(X_test)

# COMMAND ----------

MSE2 = mean_squared_error(y_test, lmpred2)

# COMMAND ----------

round(MSE2, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC **Polynomial Regression (horsepower$^3$)**

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

X = df[['horsepower']]
X_ = pd.DataFrame(PF(3).fit_transform(X))
y = df[['mpg']]

# COMMAND ----------

X_.head()

# COMMAND ----------

X_.drop(columns=0, inplace=True)

# COMMAND ----------

X_.head()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.5, random_state=1)

# COMMAND ----------

lmfit3 = LinearRegression().fit(X_train, y_train)

# COMMAND ----------

lmpred3 = lmfit3.predict(X_test)

# COMMAND ----------

MSE3 = mean_squared_error(y_test, lmpred3)

# COMMAND ----------

round(MSE3, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regressions using random state = 2

# COMMAND ----------

# MAGIC %md
# MAGIC **Simple Linear Regression**

# COMMAND ----------

X = df[['horsepower']]
y = df['mpg']

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

lmfit = LinearRegression().fit(X_train, y_train)

# COMMAND ----------

lmpred = lmfit.predict(X_test)

# COMMAND ----------

MSE = mean_squared_error(y_test, lmpred)

# COMMAND ----------

round(MSE, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC **Polynomial Regression (horsepower$^2$)**

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

X = df[['horsepower']]
X_ = pd.DataFrame(PF(2).fit_transform(X))
y = df[['mpg']]

# COMMAND ----------

X_.head()

# COMMAND ----------

X_.drop(columns=0, inplace=True)

# COMMAND ----------

X_.head()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.5, random_state=2)

# COMMAND ----------

lmfit2 = LinearRegression().fit(X_train, y_train)

# COMMAND ----------

lmpred2 = lmfit2.predict(X_test)

# COMMAND ----------

MSE2 = mean_squared_error(y_test, lmpred2)

# COMMAND ----------

round(MSE2, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC **Polynomial Regression (horsepower$^3$)**

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

X = df[['horsepower']]
X_ = pd.DataFrame(PF(3).fit_transform(X))
y = df[['mpg']]

# COMMAND ----------

X_.head()

# COMMAND ----------

X_.drop(columns=0, inplace=True)

# COMMAND ----------

X_.head()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.5, random_state=2)

# COMMAND ----------

lmfit3 = LinearRegression().fit(X_train, y_train)

# COMMAND ----------

lmpred3 = lmfit3.predict(X_test)

# COMMAND ----------

MSE3 = mean_squared_error(y_test, lmpred3)

# COMMAND ----------

round(MSE3, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC **Thus we see there is a difference in errors when we choose different training sets.**