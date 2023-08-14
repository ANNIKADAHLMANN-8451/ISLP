# Databricks notebook source
# MAGIC %md
# MAGIC In this exercise, you will further analyze the `Wage` data set considered
# MAGIC throughout this chapter.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,}'.format
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Wage.csv"
Wage = spark.read.option("header", "true").csv(url).toPandas()
Wage.set_index("SlNo", inplace=True)

str_cols = ["maritl", "race", "education", "region", "jobclass", "health", "health_ins"]
float_cols = ["logwage", "wage"]
int_cols = list(set(Wage.columns)-set(str_cols)-set(float_cols))
Wage[str_cols] = Wage[str_cols].astype(str)
Wage[float_cols] = Wage[float_cols].astype(float)
Wage[int_cols] = Wage[int_cols].astype(int)

# COMMAND ----------

Wage.head()

# COMMAND ----------

Wage.describe().round(2)

# COMMAND ----------

Wage.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Perform polynomial regression to predict `wage` using `age`. Use
# MAGIC cross-validation to select the optimal degree d for the polynomial. What degree was chosen, and how does this compare to
# MAGIC the results of hypothesis testing using ANOVA? Make a plot of
# MAGIC the resulting polynomial ft to the data.**

# COMMAND ----------

from sklearn.model_selection import KFold as KF, cross_val_score as CVS
from sklearn.linear_model import LinearRegression

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a step function to predict `wage` using `age`, and perform crossvalidation to choose the optimal number of cuts. Make a plot of
# MAGIC the ft obtained.**

# COMMAND ----------

# TODO: your response here