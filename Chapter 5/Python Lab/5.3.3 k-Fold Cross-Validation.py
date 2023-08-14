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
# MAGIC ### k-Fold Cross-Validation

# COMMAND ----------

from sklearn.model_selection import KFold as KF

# COMMAND ----------

kf = KF(n_splits=10) # k = 10

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# COMMAND ----------

MSE_all = pd.DataFrame()

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

for i in range(1,11):
    MSE = 0
    X = df[['horsepower']]
    X_ = pd.DataFrame(PF(i).fit_transform(X))
    X_.drop(columns=0, inplace=True)
    y = df[['mpg']]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_.iloc[train_index], X_.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lmfit = LinearRegression().fit(X_train, y_train)
        lmpred = lmfit.predict(X_test)
        MSE += mean_squared_error(y_test, lmpred)
    MSE_mean = MSE/10
    MSE_all = MSE_all.append([MSE_mean])

# COMMAND ----------

MSE_all.columns = [['MSE']]
MSE_all.reset_index(drop=True, inplace=True)
round(MSE_all, 2)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(MSE_all, color='green', marker='o', linestyle='dashed',
        linewidth=2, markersize=12, markerfacecolor = 'orange')
plt.title("MSE vs order of regression")
plt.xlabel("order")
plt.ylabel("MSE")

# COMMAND ----------

# MAGIC %md
# MAGIC **Some observations: First, the computation time was significantly lesser than LOOCV. Second, there is a slight difference
# MAGIC in errors between LOOCV and K-Fold CV for first order regression. This is because the k-Fold CV displays bias.**