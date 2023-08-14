# Databricks notebook source
# MAGIC %md
# MAGIC We will now consider the `Boston` housing data set.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd
from statistics import stdev
from math import sqrt

# COMMAND ----------

# import relevant data visualisaton packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas()
Boston.set_index("SlNo", inplace=True)

int_cols = ['chas', 'rad']
float_cols = list(set(Boston.columns)-set(int_cols))
Boston[float_cols] = Boston[float_cols].astype(float)
Boston[int_cols] = Boston[int_cols].astype(int)

# COMMAND ----------

Boston.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Based on this data set, provide an estimate for the population
# MAGIC mean of `medv`. Call this estimate µˆ.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **Provide an estimate of the standard error of µˆ. Interpret this
# MAGIC result.**
# MAGIC <br>
# MAGIC *Hint: We can compute the standard error of the sample mean by
# MAGIC dividing the sample standard deviation by the square root of the
# MAGIC number of observations.*

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Now estimate the standard error of µˆ using the bootstrap. How
# MAGIC does this compare to your answer from (b)?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Based on your bootstrap estimate from (c), provide a 95 % confdence interval for the mean of `medv`. Compare it to the results
# MAGIC obtained by using `Boston['medv'].std()` and the two standard
# MAGIC error rule (3.9).**
# MAGIC <br>
# MAGIC *Hint: You can approximate a 95 % confdence interval using the
# MAGIC formula [ˆµ − 2SE(ˆµ), µˆ + 2SE(ˆµ)].*

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Based on this data set, provide an estimate, µ_med, for the median
# MAGIC value of `medv` in the population.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. We now would like to estimate the standard error of µˆ_med. Unfortunately, there is no simple formula for computing the standard
# MAGIC error of the median. Instead, estimate the standard error of the
# MAGIC median using the bootstrap. Comment on your fndings.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Based on this data set, provide an estimate for the tenth percentile of `medv` in Boston census tracts. Call this quantity µˆ0.1.
# MAGIC (You can use the `np.percentile()` function.)**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Use the bootstrap to estimate the standard error of µˆ0.1. Comment on your fndings.**

# COMMAND ----------

# TODO: your response here