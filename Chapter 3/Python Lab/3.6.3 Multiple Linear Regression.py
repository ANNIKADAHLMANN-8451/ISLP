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
X1 = Boston['crim']
X2 = Boston['zn']
X3 = Boston['indus']
X4 = Boston['chas']
X5 = Boston['nox']
X6 = Boston['rm']
X7 = Boston['age']
X8 = Boston['dis']
X9 = Boston['rad']
X10 = Boston['tax']
X11 = Boston['ptratio']
X12 = Boston['black']
X13 = Boston['lstat']

model = ols("Y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13", data = Boston).fit()

# COMMAND ----------

model.summary()

# COMMAND ----------

# calculate and display variance inflation factor
Bostondf = pd.concat([X1, X2, X3, X4, X5, X6, X7, X8, \
X9, X10, X11, X12, X13], axis = 1)
vif = pd.DataFrame()
vif["Variance Inflation Factor"] = [variance_inflation_factor(Bostondf.values, i)\
for i in range(Bostondf.shape[1])]
vif["Features"] = Bostondf.columns
vif["Variance Inflation Factor"]

# COMMAND ----------

vif["Features"]

# COMMAND ----------


