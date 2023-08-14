# Databricks notebook source
# MAGIC %md
# MAGIC Apply boosting, bagging, random forests, and BART to a data set
# MAGIC of your choice. Be sure to ft the models on a training set and to
# MAGIC evaluate their performance on a test set. How accurate are the results
# MAGIC compared to simple methods like linear or logistic regression? Which
# MAGIC of these approaches yields the best performance?

# COMMAND ----------

# MAGIC %pip install --quiet mlxtend

# COMMAND ----------

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

# import custom packages
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as r2, mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_linear_regression as PLS
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Heart.csv"
df = spark.read.option("header", "true").csv(url).toPandas()
df.set_index("_c0", inplace=True)

str_cols = ["ChestPain", "Thal", "AHD"]
float_cols = ["Ca", "Oldpeak"]
int_cols = list(set(df.columns)-set(str_cols)-set(float_cols))
df[str_cols] = df[str_cols].astype(str)
df[float_cols] = df[float_cols].astype(float)
df[int_cols] = df[int_cols].astype(int)

# COMMAND ----------

df.ChestPain.value_counts()

# COMMAND ----------

df.Thal.value_counts()

# COMMAND ----------

df.AHD.value_counts()

# COMMAND ----------

df.AHD = df.AHD.map({'No': 0, 'Yes': 1})

# COMMAND ----------

df.Thal = pd.factorize(df.Thal)[0]

# COMMAND ----------

df.ChestPain = pd.factorize(df.ChestPain)[0]

# COMMAND ----------

df.head(25)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('missing values in df', fontsize=30, color='m')
plt.xlabel('features', fontsize=20, color='c')

# COMMAND ----------

df.drop(axis=0, inplace=True, columns='Ca')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('missing values in df', fontsize=30, color='m')
plt.xlabel('features', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we have no missing values in df.

# COMMAND ----------

# TODO: your response here