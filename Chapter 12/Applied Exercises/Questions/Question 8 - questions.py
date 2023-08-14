# Databricks notebook source
# MAGIC %md
# MAGIC In Section 12.2.3, a formula for calculating PVE was given in Equation 12.10. We also saw that the PVE can be obtained using the
# MAGIC `explained_variance_ratio_` attribute of a fitted `PCA()` estimator. On the `USArrests` data, calculate PVE in the two ways highlighted below.

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
# MAGIC **a. Using the `explained_variance_ratio_` output of the fitted `PCA()`
# MAGIC estimator, as was done in Section 12.2.3.**

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. By applying Equation 12.10 directly. The loadings are stored
# MAGIC as the `components_` attribute of the fitted `PCA()` estimator. Use
# MAGIC those loadings in Equation 12.10 to obtain the PVE.**

# COMMAND ----------

# TODO: your response here