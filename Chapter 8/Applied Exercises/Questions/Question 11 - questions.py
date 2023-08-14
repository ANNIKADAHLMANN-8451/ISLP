# Databricks notebook source
# MAGIC %md
# MAGIC This question uses the `Caravan` data set.

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

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Caravan.csv"
Caravan = spark.read.option("header", "true").csv(url).toPandas()
Caravan.set_index('_c0', inplace=True)
Caravan.index.names = ['Index']

str_cols = ["Purchase"]
num_cols = list(set(Caravan.columns) - set(str_cols))
Caravan[str_cols] = Caravan[str_cols].astype(str)
Caravan[num_cols] = Caravan[num_cols].astype(float)

# COMMAND ----------

Caravan.Purchase.value_counts()

# COMMAND ----------

Caravan.Purchase = Caravan.Purchase.map({'No': 0, 'Yes': 1})
Caravan.head()

# COMMAND ----------

Caravan.Purchase.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a training set consisting of the frst 1,000 observations,
# MAGIC and a test set consisting of the remaining observations.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b.  Fit a boosting model to the training set with `Purchase` as the
# MAGIC response and the other variables as predictors. Use 1,000 trees,
# MAGIC and a shrinkage value of 0.01. Which predictors appear to be
# MAGIC the most important?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Use the boosting model to predict the response on the test data.
# MAGIC Predict that a person will make a purchase if the estimated probability of purchase is greater than 20 %. Form a confusion matrix. What fraction of the people predicted to make a purchase
# MAGIC do in fact make one? How does this compare with the results
# MAGIC obtained from applying KNN or logistic regression to this data
# MAGIC set?**

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

# TODO: your response here