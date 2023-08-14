# Databricks notebook source
# MAGIC %md
# MAGIC This problem is working with a gene expression data set (Ch12Ex13.csv) that consists of 40 tissue samples with
# MAGIC measurements on 1,000 genes. The first 20 samples are from healthy patients, while the second 20 are from a diseased group.

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

# MAGIC %md
# MAGIC **a. Load in the data using `pd.read_csv()`. You will need to select
# MAGIC `header = None`.**

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Ch12Ex13.csv"
df = spark.read.option("header", "false").csv(url).toPandas().astype(float)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Apply hierarchical clustering to the samples using correlation-based distance, and plot the dendrogram. Do the genes separate
# MAGIC the samples into the two groups? Do your results depend on the
# MAGIC type of linkage used?**

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import pdist

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Your collaborator wants to know which genes difer the most
# MAGIC across the two groups. Suggest a way to answer this question,
# MAGIC and apply it here.**

# COMMAND ----------

from sklearn.decomposition import PCA

# COMMAND ----------

# TODO: your response here