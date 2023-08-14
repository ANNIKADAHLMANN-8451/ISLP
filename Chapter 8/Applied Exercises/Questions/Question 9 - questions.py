# Databricks notebook source
# MAGIC %md
# MAGIC This problem involves the `OJ` data set.

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
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/OJ.csv"
df = spark.read.option("header", "true").csv(url).toPandas()
df.set_index("SlNo", inplace=True)

str_cols = ["Purchase", "Store7", "education", "region", "jobclass", "health", "health_ins"]
float_cols = ["PriceCH", "PriceMM", "DiscCH", "DiscMM", "LoyalCH", "SalePriceMM", "SalePriceCH", "PriceDiff", "PctDiscMM", "PctDiscCH", "ListPriceDiff"]
int_cols = list(set(df.columns)-set(str_cols)-set(float_cols))
df[str_cols] = df[str_cols].astype(str)
df[float_cols] = df[float_cols].astype(float)
df[int_cols] = df[int_cols].astype(int)

# COMMAND ----------

df.head()

# COMMAND ----------

df.Purchase = df.Purchase.map(lambda x: 1 if x=='CH' else 0)
df.Store7 = df.Store7.map({'No': 0, 'Yes': 1})

# COMMAND ----------

df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a tree to the training data, with `Purchase` as the response
# MAGIC and the other variables as predictors. What is the training error
# MAGIC rate?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. How many
# MAGIC terminal nodes does the tree have?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Use cross-validation on the training set in order to determine
# MAGIC the optimal tree size.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **i. Produce a pruned tree corresponding to the optimal tree size
# MAGIC obtained using cross-validation. If cross-validation does not lead
# MAGIC to selection of a pruned tree, then create a pruned tree with fve
# MAGIC terminal nodes.**

# COMMAND ----------

# TODO: your response here