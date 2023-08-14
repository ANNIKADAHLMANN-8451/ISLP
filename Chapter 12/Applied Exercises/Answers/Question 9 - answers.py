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

hc_complete = linkage(y=USArrests, method='complete', metric='euclidean')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('complete linkage', fontsize=30, color='m')
plt.xlabel('index', fontsize=20, color='c')
plt.ylabel('distance', fontsize=20, color='c')
axes = USArrests.index.values
dendrogram(hc_complete, leaf_rotation=90., leaf_font_size=15., show_leaf_counts=False, labels=axes)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **b.  Cut the dendrogram at a height that results in three distinct
# MAGIC clusters. Which states belong to which clusters?**

# COMMAND ----------

hc_cut3 = pd.DataFrame(cut_tree(hc_complete, n_clusters=3))
hc_cut3.columns = ['cluster']
hc_cut3.index = USArrests.index
hc_cut3.head()

# COMMAND ----------

hc_complete_3 = linkage(y=hc_cut3, method='complete', metric='euclidean')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('complete linkage', fontsize=30, color='m')
plt.xlabel('index', fontsize=20, color='c')
plt.ylabel('distance', fontsize=20, color='c')
axes = USArrests.index.values
dendrogram(hc_complete_3, leaf_rotation=90., leaf_font_size=15., show_leaf_counts=False, labels=axes)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation one.**

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import scale

# COMMAND ----------

df = pd.DataFrame(scale(USArrests))
df.describe()

# COMMAND ----------

hc_complete_scaled = linkage(y=df, method='complete', metric='euclidean')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('complete linkage', fontsize=30, color='m')
plt.xlabel('index', fontsize=20, color='c')
plt.ylabel('distance', fontsize=20, color='c')
axes = USArrests.index.values
dendrogram(hc_complete_scaled, leaf_rotation=90., leaf_font_size=15., show_leaf_counts=False, labels=axes)
plt.show()

# COMMAND ----------

hc_cut3_scaled = pd.DataFrame(cut_tree(hc_complete_scaled, n_clusters=3))
hc_cut3_scaled.columns = ['cluster']
hc_cut3_scaled.index = USArrests.index
hc_cut3_scaled.head()

# COMMAND ----------

hc_complete_3_scaled = linkage(y=hc_cut3_scaled, method='complete', metric='euclidean')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('complete linkage', fontsize=30, color='m')
plt.xlabel('index', fontsize=20, color='c')
plt.ylabel('distance', fontsize=20, color='c')
axes = USArrests.index.values
dendrogram(hc_complete_3_scaled, leaf_rotation=90., leaf_font_size=15., show_leaf_counts=False, labels=axes)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **d. What efect does scaling the variables have on the hierarchical
# MAGIC clustering obtained? In your opinion, should the variables be
# MAGIC scaled before the inter-observation dissimilarities are computed?
# MAGIC Provide a justifcation for your answer.**

# COMMAND ----------

# MAGIC %md
# MAGIC *Scaling variables increases the number of clusters and when introducing cuts of 3 clusters, it changes the composition of clusters.<br>Scaling of variables before computing inter-observation dissimilarities are computed is desirable because the metric used to count different features in a data set. E.g. In the `USArrests` data set, `UrbanPop` data is likely to be measured in a different way to `Assault`. So, scaling ensures like data are being compared.*