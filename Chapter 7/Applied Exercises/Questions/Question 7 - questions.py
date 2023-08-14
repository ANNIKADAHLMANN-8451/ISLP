# Databricks notebook source
# MAGIC %md
# MAGIC The `Wage` data set contains a number of other features not explored
# MAGIC in this chapter, such as marital status (`maritl`), job class (`jobclass`),
# MAGIC and others. Explore the relationships between some of these other
# MAGIC predictors and `wage`, and use non-linear ftting techniques in order to
# MAGIC ft fexible models to the data. Create plots of the results obtained,
# MAGIC and write a summary of your fndings.

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing

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

# TODO: your response here