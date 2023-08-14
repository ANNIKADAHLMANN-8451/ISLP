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

hc_complete = linkage(y=df, method='complete', metric='correlation')

plt.xkcd()
plt.figure(figsize=(25, 10))
dend = dendrogram(hc_complete, color_threshold=3)

# COMMAND ----------

hc_average = linkage(y=df, method='average', metric='correlation')

plt.xkcd()
plt.figure(figsize=(25, 10))
dend = dendrogram(hc_average, color_threshold=3)

# COMMAND ----------

hc_single = linkage(y=df, method='single', metric='correlation')

plt.xkcd()
plt.figure(figsize=(25, 10))
dend = dendrogram(hc_single, color_threshold=3)

# COMMAND ----------

# MAGIC %md
# MAGIC Yes, the genes separate the samples into two groups. And yes, my results depend on the type of linkage used.

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Your collaborator wants to know which genes difer the most
# MAGIC across the two groups. Suggest a way to answer this question,
# MAGIC and apply it here.**

# COMMAND ----------

from sklearn.decomposition import PCA

# COMMAND ----------

pca = PCA(n_components=40)
pca_data = pca.fit_transform(df)
principaldf = pd.DataFrame(data = pca_data)
principaldf.head()

# COMMAND ----------

loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df.columns, columns=principaldf.columns)
loadings_df.head()

# COMMAND ----------

loadings_total = pd.DataFrame()

for row in principaldf.index:
    row_sum = np.sum(principaldf.loc[row])
    loadings_total = loadings_total.append(pd.DataFrame([row_sum]))

loadings_total.columns = ['total loadings']
loadings_total.reset_index(drop=True, inplace=True)
loadings_total.head()

# COMMAND ----------

most_different_genes = pd.DataFrame(np.argsort(np.abs(loadings_total['total loadings'])))
most_different_genes.columns = ['most different genes']
most_different_genes

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the 10 most different genes are: 5, 179, 615, 380, 805, 190, 254, 558, 420, 990.