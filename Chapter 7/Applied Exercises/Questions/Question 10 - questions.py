# Databricks notebook source
# MAGIC %md
# MAGIC This question relates to the `College` data set.

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
from sklearn.metrics import r2_score, mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_linear_regression as PLS
from sklearn.preprocessing import scale
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/College.csv"
College = spark.read.option("header", "true").csv(url).toPandas()
College.set_index("_c0", inplace=True)

str_cols = ["private"]
float_cols = ["S.F.Ratio"]
int_cols = list(set(College.columns)-set(str_cols)-set(float_cols))
College[str_cols] = College[str_cols].astype(str)
College[float_cols] = College[float_cols].astype(float)
College[int_cols] = College[int_cols].astype(int)

# COMMAND ----------

College.head()

# COMMAND ----------

College.info()

# COMMAND ----------

College = pd.get_dummies(data=College, columns=['Private'], drop_first=True)

# COMMAND ----------

College.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Split the data into a training set and a test set. Using out-of-state
# MAGIC tuition as the response and the other variables as the predictors,
# MAGIC perform forward stepwise selection on the training set in order
# MAGIC to identify a satisfactory model that uses just a subset of the
# MAGIC predictors.**

# COMMAND ----------

from sklearn.model_selection import train_test_split
from pygam import LinearGAM
from pygam.terms import gen_edge_knots
from sklearn.ensemble.partial_dependence import plot_partial_dependence

# COMMAND ----------

# TODO: your response here