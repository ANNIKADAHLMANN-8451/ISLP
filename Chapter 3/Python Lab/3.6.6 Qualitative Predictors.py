# Databricks notebook source
# import statistical tools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy.contrasts import Treatment

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load data; visualisation same as Section 3.6.3
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Carseats.csv"
Carseats = spark.read.option("header", "true").csv(url).toPandas()
Carseats.set_index('SlNo', inplace=True)

str_cols = ["ShelveLoc", "Urban", "US"]
num_cols = ["Sales", "CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]
Carseats[str_cols] = Carseats[str_cols].astype(str)
Carseats[num_cols] = Carseats[num_cols].astype(float)

# COMMAND ----------

# view and visualise data
Carseats.head()

# COMMAND ----------

sns.pairplot(Carseats, kind = "reg")

# COMMAND ----------

# perform regression
Y = Carseats['Sales']
X1 = Carseats['CompPrice']
X2 = Carseats['Income']
X3 = Carseats['Advertising']
X4 = Carseats['Population']
X5 = Carseats['Price']
X6 = Carseats['ShelveLoc']
X7 = Carseats['Age']
X8 = Carseats['Education']
X9 = Carseats['Urban']
X10 = Carseats['US']

model = ols("Y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X2:X3+X5:X7", data = Carseats).fit()

# COMMAND ----------

model.summary()

# COMMAND ----------

# understanding dummy variables
levels = [1, 2, 3]
contrast = Treatment(reference=0).code_without_intercept(levels)

# COMMAND ----------

contrast.matrix

# COMMAND ----------

