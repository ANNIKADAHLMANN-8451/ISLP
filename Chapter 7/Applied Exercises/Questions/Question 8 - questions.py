# Databricks notebook source
# MAGIC %md
# MAGIC Fit some of the non-linear models investigated in this chapter to the
# MAGIC `Auto` data set. Is there evidence for non-linear relationships in this
# MAGIC data set? Create some informative plots to justify your answer.

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
import seaborn as sns; sns.set(style="ticks", color_codes=True)
%matplotlib inline

# COMMAND ----------

url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["name"]
num_cols = list(set(Auto.columns)-set(str_cols))
Auto[str_cols] = Auto[str_cols].astype(str)
Auto[num_cols] = Auto[num_cols].astype(float)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(Auto[['horsepower']]=='?', yticklabels=False, xticklabels=False, cmap='viridis')
plt.title("missing values in Auto['horsepower']", fontsize=30, color='m')

# COMMAND ----------

Auto.loc[Auto.horsepower=="?"]

# COMMAND ----------

Auto.shape

# COMMAND ----------

Auto.drop(index=[32, 126, 330, 336, 354], inplace=True)

# COMMAND ----------

Auto.shape

# COMMAND ----------

# MAGIC %md
# MAGIC *So,I have deleted the rows containing erroneous values of horsepower.*

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(Auto[['horsepower']]=='?', yticklabels=False, xticklabels=False, cmap='viridis')
plt.title("missing values in Auto['horsepower']", fontsize=30, color='m')

# COMMAND ----------

# MAGIC %md
# MAGIC *See, no missing values!*

# COMMAND ----------

Auto.corr()

# COMMAND ----------

from sklearn.model_selection import KFold as KF, cross_val_score as CVS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# TODO: your response here