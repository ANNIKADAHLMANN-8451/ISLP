# Databricks notebook source
# import statistical tools
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# load and visualise data
# import dataset and preprocess
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index('SlNo', inplace=True)
Boston.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.pairplot(Boston)
plt.show()

# COMMAND ----------

# perform regression
Y = Boston['medv']
X1 = Boston['lstat']
X2 = Boston['age']
model = ols('Y~X1*X2', data = Boston).fit()

# COMMAND ----------

model.summary()

# COMMAND ----------

