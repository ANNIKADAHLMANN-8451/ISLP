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

USArrests.head()

# COMMAND ----------

list(USArrests)

# COMMAND ----------

USArrests.mean()

# COMMAND ----------

USArrests.var()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Principal Components Analysis

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
df = pd.DataFrame(StandardScaler().fit_transform(USArrests))

# COMMAND ----------

df.columns = USArrests.columns
df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

df_mean = pd.DataFrame(df.mean(), columns=['mean'])
df_std = pd.DataFrame(df.std(), columns=['standard deviation'])
df_moments = pd.concat([df_mean, df_std], axis=1)
df_moments

# COMMAND ----------

pca = PCA(n_components=4)
pca_data = pca.fit_transform(df)
principalDf = pd.DataFrame(data = pca_data, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
principalDf.head()

# COMMAND ----------

principalDf.info()

# COMMAND ----------

loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df.columns, columns=principalDf.columns)
loadings_df

# COMMAND ----------

principalDf.shape

# COMMAND ----------

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0, shrinkC=0, shrinkD=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(principalDf['PC1'], principalDf['PC2'], alpha=0.25, s=200, color='green')
plt.title('principal components', fontsize=30, color='m')
plt.xlabel('principal component 1', fontsize=20, color='c')
plt.ylabel('principal component 2', fontsize=20, color='c')
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 2 * np.sqrt(length)
    arrow = plt.arrow(0, 0, pca.mean_[0] + v[0], pca.mean_[1] + v[1], label='mylabel', 
                      width=0.09, facecolor='orange', edgecolor='orange', alpha=0.5, )

# COMMAND ----------

PSTD = np.sqrt(pca.explained_variance_)
PSTD

# COMMAND ----------

PEV = pca.explained_variance_
PEV

# COMMAND ----------

PVE = pca.explained_variance_ratio_
PVE

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(np.cumsum(PVE), lw=5.0, ls='-.', color='g', marker='o', markersize=15, markerfacecolor='orange')
plt.xlabel('principal component', fontsize=20, color='c')
plt.ylabel('cumulative proportion of variance explained', fontsize=20, color='c')
plt.title('principal components cumulative explained variance', fontsize=30, color='m')

# COMMAND ----------

