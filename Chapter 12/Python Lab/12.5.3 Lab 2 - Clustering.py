# Databricks notebook source
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

np.random.seed(2)
x1 = pd.DataFrame(np.random.normal(size=50), columns=['col1'])
x2 = pd.DataFrame(np.random.normal(size=50), columns=['col2'])
x = pd.concat([x1, x2], axis=1)
x

# COMMAND ----------

x.col1.iloc[0:24] += 3
x.col2.iloc[0:24] -= 4
x

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.5.1 $K$-means clustering

# COMMAND ----------

# MAGIC %md
# MAGIC **$K$=2**

# COMMAND ----------

from sklearn.cluster import KMeans as KM

# COMMAND ----------

km_out = KM(n_clusters=2, n_init=20).fit(x)

# COMMAND ----------

km_labels = km_out.labels_
km_labels

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x.col1[km_labels==0], x.col2[km_labels==0], color='green', s=500, alpha=0.5)
plt.scatter(x.col1[km_labels==1], x.col2[km_labels==1], color='orange', s=500, alpha=0.5)
plt.xlabel('col1', fontsize=20, color='c')
plt.ylabel('col2', fontsize=20, color='c')
plt.title('K-means clustering results with K=2', fontsize=30, color='m')

# COMMAND ----------

# MAGIC %md
# MAGIC **$K$=3**

# COMMAND ----------

np.random.seed(4) # this isn't the same as the seed in R mentioned in book. Nonetheless, I use the same seed here
km_out = KM(n_clusters=3, n_init=20).fit(x)
km_labels = km_out.labels_
km_labels

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x.col1[km_labels==0], x.col2[km_labels==0], color='green', s=500, alpha=0.5)
plt.scatter(x.col1[km_labels==1], x.col2[km_labels==1], color='orange', s=500, alpha=0.5)
plt.scatter(x.col1[km_labels==2], x.col2[km_labels==2], color='blue', s=500, alpha=0.5)
plt.xlabel('col1', fontsize=20, color='c')
plt.ylabel('col2', fontsize=20, color='c')
plt.title('K-means clustering results with K=3', fontsize=30, color='m')

# COMMAND ----------

k_cluster_means = pd.DataFrame(km_out.cluster_centers_, columns=['col1', 'col2'])
k_cluster_means

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.5.2 Hierarchial clustering

# COMMAND ----------

from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree

# COMMAND ----------

hc_complete = linkage(y=x, method='complete')
hc_average = linkage(y=x, method='average')
hc_single = linkage(y=x, method='single')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('complete linkage', fontsize=30, color='m')
plt.xlabel('index', fontsize=20, color='c')
plt.ylabel('distance', fontsize=20, color='c')
dendrogram(hc_complete, leaf_rotation=90., leaf_font_size=15., show_leaf_counts=True)
plt.show()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('average linkage', fontsize=30, color='m')
plt.xlabel('index', fontsize=20, color='c')
plt.ylabel('distance', fontsize=20, color='c')
dendrogram(hc_average, leaf_rotation=90., leaf_font_size=15., show_leaf_counts=True)
plt.show()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('single linkage', fontsize=30, color='m')
plt.xlabel('index', fontsize=20, color='c')
plt.ylabel('distance', fontsize=20, color='c')
dendrogram(hc_single, leaf_rotation=90., leaf_font_size=15., show_leaf_counts=True)
plt.show()

# COMMAND ----------

cut_tree(hc_complete, n_clusters=2).T

# COMMAND ----------

cut_tree(hc_average, n_clusters=2).T

# COMMAND ----------

cut_tree(hc_single, n_clusters=2).T

# COMMAND ----------

cut_tree(hc_single, n_clusters=4).T

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

xsc = StandardScaler().fit_transform(x)
xsc

# COMMAND ----------

hc_complete_xsc = linkage(y=xsc, method='complete')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('complete linkage - scaled data', fontsize=30, color='m')
plt.xlabel('index', fontsize=20, color='c')
plt.ylabel('distance', fontsize=20, color='c')
dendrogram(hc_complete_xsc, leaf_rotation=90., leaf_font_size=15., show_leaf_counts=True)
plt.show()

# COMMAND ----------

