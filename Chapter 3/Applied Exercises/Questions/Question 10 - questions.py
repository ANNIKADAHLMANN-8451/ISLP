# Databricks notebook source
# MAGIC %md
# MAGIC This question should be answered using the `Carseats` data set.

# COMMAND ----------

# import statistical tools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# load data; visualisation same as Section 3.6.3
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Carseats.csv"
CarSeats = spark.read.option("header", "true").csv(url).toPandas()
CarSeats.set_index('SlNo', inplace=True)

str_cols = ["ShelveLoc", "Urban", "US"]
num_cols = ["Sales", "CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]
CarSeats[str_cols] = CarSeats[str_cols].astype(str)
CarSeats[num_cols] = CarSeats[num_cols].astype(float)

# COMMAND ----------

CarSeats.head()

# COMMAND ----------

list(CarSeats)

# COMMAND ----------

CarSeats.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Fit a multiple regression model to predict `Sales` using `Price`, `Urban`, and `US`.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Provide an interpretation of each coefcient in the model. Be careful—some of the variables in the model are qualitative!**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Write out the model in equation form, being careful to handle the qualitative variables properly.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. For which of the predictors can you reject the null hypothesis?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. On the basis of your response to the previous question, fit a smaller model that only uses the predictors for which there is evidence of association with the outcome.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. How well do the models in (a) and (e) ft the data?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Using the model from (e), obtain 95 % confdence intervals for the coefcient(s).**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Is there evidence of outliers or high leverage observations in the model from (e)?**

# COMMAND ----------

# TODO: your response here