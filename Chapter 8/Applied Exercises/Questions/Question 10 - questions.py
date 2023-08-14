# Databricks notebook source
# MAGIC %md
# MAGIC We now use boosting to predict `Salary` in the `Hitters` data set.

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
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Hitters.csv"
Hitters = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["Names", "NewLeague", "League", "Division"]
num_cols = list(set(Hitters.columns) - set(str_cols))
Hitters["Salary"] = np.where(Hitters["Salary"] == "NA", np.nan, Hitters["Salary"])
Hitters[str_cols] = Hitters[str_cols].astype(str)
Hitters[num_cols] = Hitters[num_cols].astype(float)

# COMMAND ----------

Hitters.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Remove the observations for whom the salary information is
# MAGIC unknown, and then log-transform the salaries.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Create a training set consisting of the frst 200 observations, and
# MAGIC a test set consisting of the remaining observations.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Perform boosting on the training set with 1,000 trees for a range
# MAGIC of values of the shrinkage parameter λ. Produce a plot with
# MAGIC diferent shrinkage values on the x-axis and the corresponding
# MAGIC training set MSE on the y-axis.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Produce a plot with diferent shrinkage values on the x-axis and
# MAGIC the corresponding test set MSE on the y-axis.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Compare the test MSE of boosting to the test MSE that results
# MAGIC from applying two of the regression approaches seen in
# MAGIC Chapters 3 and 6.**

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Which variables appear to be the most important predictors in
# MAGIC the boosted model?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Now apply bagging to the training set. What is the test set MSE
# MAGIC for this approach?**

# COMMAND ----------

# TODO: your response here