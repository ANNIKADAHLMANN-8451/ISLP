# Databricks notebook source
# MAGIC %md
# MAGIC This question involves the use of multiple linear regression on the `Auto` data set.

# COMMAND ----------

# general imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# COMMAND ----------

# import data visualisation tools
import seaborn as sns

# COMMAND ----------

# import and preprocess data
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

list(Auto)

# COMMAND ----------

Auto = Auto.drop(Auto.index[[32, 126, 330, 336, 354]]) # removing rows containing "?". This is the easy way out. Such missing values need to be explored first in a real life situation.

# COMMAND ----------

Auto.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Produce a scatterplot matrix which includes all of the variables in the data set.**

# COMMAND ----------

# TODO: your answer here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Compute the matrix of correlations between the variables using the `DataFrame.corr()` method.**

# COMMAND ----------

# TODO: your answer here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Use the `sm.OLS()` function to perform a multiple linear regression with `mpg` as the response and all other variables except name as the predictors. Use the `.summary()` function to print the results. Comment on the output. For instance:**
# MAGIC   - i. Is there a relationship between the predictors and the response? Use the `anova_lm()` function from `statsmodels` to answer this question.
# MAGIC   - ii. Which predictors appear to have a statistically signifcant relationship to the response?
# MAGIC   - iii. What does the coefcient for the year variable suggest?

# COMMAND ----------

# TODO: your answer here

# COMMAND ----------

# MAGIC %md
# MAGIC **e and f. Fit some models with interactions as described in the lab. Do any interactions appear to be statistically signifcant? Try a few different transformations of the variables, such as log, square root, raising to powers. Comment on your findings.**

# COMMAND ----------

# TODO: your answer here