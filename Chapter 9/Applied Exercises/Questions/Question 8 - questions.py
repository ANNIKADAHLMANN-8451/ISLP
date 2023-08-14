# Databricks notebook source
# MAGIC %md
# MAGIC This problem involves the OJ data set.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/OJ.csv"
OJ = spark.read.option("header", "true").csv(url).toPandas()
OJ.set_index("SlNo", inplace=True)

str_cols = ["Purchase", "Store7", "education", "region", "jobclass", "health", "health_ins"]
float_cols = ["PriceCH", "PriceMM", "DiscCH", "DiscMM", "LoyalCH", "SalePriceMM", "SalePriceCH", "PriceDiff", "PctDiscMM", "PctDiscCH", "ListPriceDiff"]
int_cols = list(set(OJ.columns)-set(str_cols)-set(float_cols))
OJ[str_cols] = OJ[str_cols].astype(str)
OJ[float_cols] = OJ[float_cols].astype(float)
OJ[int_cols] = OJ[int_cols].astype(int)

# COMMAND ----------

OJ.Purchase = pd.factorize(OJ.Purchase)[0]
OJ.Store7 = pd.factorize(OJ.Store7)[0]
OJ.head()

# COMMAND ----------

OJ.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a training set containing a random sample of 800
# MAGIC observations, and a test set containing the remaining
# MAGIC observations.**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a support vector classifer to the training data using
# MAGIC `C = 0.01`, with `Purchase` as the response and the other variables
# MAGIC as predictors. How many support points are there?**

# COMMAND ----------

from sklearn.svm import SVC

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. What are the training and test error rates?**

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Use cross-validation to select an optimal C. Consider values in
# MAGIC the range 0.01 to 10.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Compute the training and test error rates using this new value
# MAGIC for C.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Repeat parts (b) through (e) using a support vector machine
# MAGIC with a radial kernel. Use the default value for gamma.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Repeat parts (b) through (e) using a support vector machine
# MAGIC with a polynomial kernel. Set degree = 2.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **h.  Overall, which approach seems to give the best results on this
# MAGIC data?**

# COMMAND ----------

# TODO: your response here