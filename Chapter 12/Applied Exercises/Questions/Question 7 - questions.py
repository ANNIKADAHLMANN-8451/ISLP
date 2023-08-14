# Databricks notebook source
# MAGIC %md
# MAGIC In this chapter, we mentioned the use of correlation-based distance
# MAGIC and Euclidean distance as dissimilarity measures for hierarchical clustering. It turns out that these two measures are almost equivalent: if
# MAGIC each observation has been centered to have mean zero and standard
# MAGIC deviation one, and if we let rij denote the correlation between the ith
# MAGIC and jth observations, then the quantity 1 − rij is proportional to the
# MAGIC squared Euclidean distance between the ith and jth observations.
# MAGIC <br>
# MAGIC <br>
# MAGIC On the `USArrests` data, show that this proportionality holds.
# MAGIC <br>
# MAGIC <br>
# MAGIC *Hint: The Euclidean distance can be calculated using the* `pairwise_distances()` *function from the* `sklearn.metrics` *module, and* `pairwise_distances()` *correlations can be calculated using the* `np.corrcoef()` *function.*

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

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/USArrests.csv"
USArrests = spark.read.option("header", "true").csv(url).toPandas()
USArrests.rename(columns={'_c0': 'Index'}, inplace=True)
USArrests.set_index("Index", inplace=True)

int_cols = ["Assault", "UrbanPop"]
float_cols = ["Murder", "Rape"]
USArrests[int_cols] = USArrests[int_cols].astype(int)
USArrests[float_cols] = USArrests[float_cols].astype(float)

# COMMAND ----------

# TODO: your response here