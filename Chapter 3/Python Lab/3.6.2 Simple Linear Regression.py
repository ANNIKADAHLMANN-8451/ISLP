# Databricks notebook source
# MAGIC %md
# MAGIC This pathway maps to the [Simple Linear Regression](https://degreed.com/pathway/k9oy6oy4px/pathway?newWindow=true) Degreed pathway in the [Essentials of Machine Learning (EML) training](https://degreed.com/plan/3003997?newWindow=true). Use these code stubs to get hands-on experience with the exercises relevant to this section of the course. These exercises come indirectly from the [Introduction to Statistical Learning (ISL)](https://www.statlearning.com/) and were converted to Python. Clone [the adapted repo](TODO) to run and modify the below code stubs in your own environment.

# COMMAND ----------

# import statistical tools
import numpy as np
import pandas as pd
import sklearn
from statsmodels.formula.api import ols
import statsmodels as sm
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from statsmodels.stats.outliers_influence import summary_table

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import dataset and preprocess
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index('SlNo', inplace=True)

# COMMAND ----------

# basic exploration of data
Boston.head()

# COMMAND ----------

Boston.corr()

# COMMAND ----------

# fit model through linear regression
Y = Boston['medv']
X = Boston['lstat']
model = ols("Y~X", data = Boston).fit()

# COMMAND ----------

model.summary()

# COMMAND ----------

# predict the model
dt = summary_table(model, alpha = 0.5)[1]
Y_prd = dt[:, 2]
Yprd_ci_lower, Yprd_ci_upper = dt[:, 6:8].T
pd.DataFrame(np.column_stack([Y_prd, Yprd_ci_lower, Yprd_ci_upper])).head()

# COMMAND ----------

# plot graph with regression line
plt.xkcd()
plt.figure(figsize = (25, 10))
plt.figure(1).add_subplot(121)
print(sns.regplot(X, Y, data = model, color = 'g'))
plt.title("Linear Model")

plt.figure(figsize = (25, 10))
plt.figure(2).add_subplot(122)
print(sns.residplot(X, Y, lowess = True, color = 'r'))
plt.title("Non-Linear Model")

# COMMAND ----------

