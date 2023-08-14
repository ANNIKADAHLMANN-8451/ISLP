# Databricks notebook source
# import statistical tools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# load and visualise data
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

# perform regression without transformation
Y = Boston['medv']
X1 = Boston['lstat']
model1 = ols('Y~X1+I(pow(X1, 2))', data = Boston).fit()

# COMMAND ----------

model1.summary()

# COMMAND ----------

# perform regression with transformation
X2 = lambda X1 : pow(X1,2) # X2 is a lambda function that raises each element of X1 to the power of 2
model2 = ols('Y~X1+I(pow(X1, 2))', data = Boston).fit()

# COMMAND ----------

model2.summary()

# COMMAND ----------

# perform anova between model1 and model2
anova_table = sm.stats.anova_lm(model1, model2) # results not ideal. Need to refine this
anova_table

# COMMAND ----------

