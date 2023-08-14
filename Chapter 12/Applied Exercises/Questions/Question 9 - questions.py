# Databricks notebook source
# MAGIC %md
# MAGIC Consider the `USArrests` data. We will now perform hierarchical clustering on the states.

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

# MAGIC %md
# MAGIC **a. Using hierarchical clustering with complete linkage and
# MAGIC Euclidean distance, cluster the states.**

# COMMAND ----------

from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b.  Cut the dendrogram at a height that results in three distinct
# MAGIC clusters. Which states belong to which clusters?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation one.**

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import scale

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. What efect does scaling the variables have on the hierarchical
# MAGIC clustering obtained? In your opinion, should the variables be
# MAGIC scaled before the inter-observation dissimilarities are computed?
# MAGIC Provide a justifcation for your answer.**

# COMMAND ----------

# TODO: your response here