# Databricks notebook source
# MAGIC %md
# MAGIC  In the lab, a classifcation tree was applied to the `Carseats` data set after converting `Sales` into a qualitative response variable. Now we will
# MAGIC seek to predict `Sales` using regression trees and related approaches,
# MAGIC treating the response as a quantitative variable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

# COMMAND ----------

# import data visualisation packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Carseats.csv"
df = spark.read.option("header", "true").csv(url).toPandas()
df.set_index("SlNo", inplace=True)

str_cols = ["ShelveLoc", "Urban", "US"]
float_cols = ["Sales"]
int_cols = list(set(df.columns)-set(str_cols)-set(float_cols))
df[str_cols] = df[str_cols].astype(str)
df[float_cols] = df[float_cols].astype(float)
df[int_cols] = df[int_cols].astype(int)


df.Sales = df.Sales.map(lambda x: 0 if x<=8 else 1)
df.ShelveLoc = pd.factorize(df.ShelveLoc)[0]
df.Urban = df.Urban.map({'No':0, 'Yes':1})
df.US = df.US.map({'No':0, 'Yes':1})

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Split the data set into a training set and a test set.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a regression tree to the training set. Plot the tree, and interpret the results. What test MSE do you obtain?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Use cross-validation in order to determine the optimal level of
# MAGIC tree complexity. Does pruning the tree improve the test MSE?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Use the bagging approach in order to analyze this data. What
# MAGIC test MSE do you obtain? Use the `feature\_importance\_ values` to
# MAGIC determine which variables are most important.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Use random forests to analyze this data. What test MSE do
# MAGIC you obtain? Use the `feature\_importance\_ values` to determine
# MAGIC which variables are most important. Describe the efect of m, the
# MAGIC number of variables considered at each split, on the error rate
# MAGIC obtained.**

# COMMAND ----------

# TODO: your response here