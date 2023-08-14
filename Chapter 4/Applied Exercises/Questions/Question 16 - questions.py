# Databricks notebook source
# MAGIC %md
# MAGIC  Using the `Boston` data set, fit classifcation models in order to predict
# MAGIC whether a given suburb has a crime rate above or below the median.
# MAGIC Explore logistic regression, LDA, naive Bayes, and KNN models using
# MAGIC various subsets of the predictors. Describe your fndings.
# MAGIC <br>
# MAGIC *Hint: You will have to create the response variable yourself, using the
# MAGIC variables that are contained in the `Boston` data set.*

# COMMAND ----------

# import statistical packages
import numpy as np
import pandas as pd

# COMMAND ----------

# import data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index('SlNo', inplace=True)

# COMMAND ----------

Boston.head()

# COMMAND ----------

# TODO: your response here