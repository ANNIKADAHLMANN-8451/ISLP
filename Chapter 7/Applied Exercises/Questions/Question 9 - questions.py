# Databricks notebook source
# MAGIC %md
# MAGIC This question uses the variables `dis` (the weighted mean of distances
# MAGIC to fve Boston employment centers) and `nox` (nitrogen oxides concentration in parts per 10 million) from the `Boston` data. We will treat
# MAGIC `dis` as the predictor and `nox` as the response.

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

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index("SlNo", inplace=True)

# COMMAND ----------

Boston = pd.get_dummies(Boston, columns =['chas'], drop_first=True)

# COMMAND ----------

Boston.head()

# COMMAND ----------

Boston.shape

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Use the `LinearRegression()` function from `sklearn` module to ft a
# MAGIC cubic polynomial regression to predict `nox` using `dis`. Report the
# MAGIC regression output, and plot the resulting data and polynomial
# MAGIC fts.**

# COMMAND ----------

X = Boston['dis']
y = Boston['nox']

# COMMAND ----------

from sklearn.model_selection import KFold as KF, cross_val_score as CVS
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Plot the polynomial fts for a range of diferent polynomial
# MAGIC degrees (say, from 1 to 10), and report the associated residual
# MAGIC sum of squares.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Use the `LinearRegression()` function from the `sklearn` module to fit a regression spline to predict `nox` using `dis`. Report the output for
# MAGIC the fit using four degrees of freedom. How did you choose the
# MAGIC knots? Plot the resulting fit.**

# COMMAND ----------

from patsy import dmatrix

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now fit a regression spline for a range of degrees of freedom, and
# MAGIC plot the resulting fits and report the resulting RSS. Describe the
# MAGIC results obtained.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Perform cross-validation or another approach in order to select
# MAGIC the best degrees of freedom for a regression spline on this data.
# MAGIC Describe your results.**

# COMMAND ----------

from sklearn.pipeline import make_pipeline
X = Boston.dis.values.reshape(-1, 1)
y = Boston.dis.values

# COMMAND ----------

# TODO: your response here