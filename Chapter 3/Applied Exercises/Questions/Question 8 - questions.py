# Databricks notebook source
# MAGIC %md
# MAGIC This question involves the use of simple linear regression on the `Auto` data set.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Imports and load dataset that you'll need to answer the below applied questions.

# COMMAND ----------

# import generic tools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

int_cols = ["cylinders", "horsepower", "weight", "year", "origin"]
float_cols = ["mpg", "displacement", "acceleration"]
str_cols = ["name"]
Auto[int_cols] = Auto[int_cols].astype(int)
Auto[float_cols] = Auto[float_cols].astype(float)
Auto[str_cols] = Auto[str_cols].astype(str)

# COMMAND ----------

Auto.head()

# COMMAND ----------

Auto = Auto.drop(Auto.index[[32, 126, 330, 336, 354]]) # removing rows containing "?". This is the easy way out. Such missing values need to be explored first in a real life situation.

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Use the `sm.OLS()` function to perform a simple linear regression with `mpg` as the response and `horsepower` as the predictory. Use the `.summary()` function to print the results. Comment on the output. Consider using `stats.linregress` to gather the necessary information. For example:**
# MAGIC   - Is there a relationship between the predictor and the response?
# MAGIC   - How strong is the relationship between the predictor and the response?
# MAGIC   - Is the relationship between the predictor and the response positive or negative?
# MAGIC   - What is the predicted `mpg` associated with a `horsepower` of 98? What are the associated 95 % confdence and prediction intervals?

# COMMAND ----------

# TODO: your answer here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Plot the response and the predictory in a new `matplotlib` figure. Use the `sns.lmplot()` method to display the least squares regression line.**

# COMMAND ----------

# TODO: your answer here