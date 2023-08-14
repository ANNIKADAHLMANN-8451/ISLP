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

# MAGIC %md
# MAGIC Comparing measures

# COMMAND ----------

from sklearn.preprocessing import scale

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
df = pd.DataFrame(scale(USArrests))
df.columns = USArrests.columns
df.head()

# COMMAND ----------

df.describe().round(4)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.distplot(df.Murder, bins=30, kde_kws={'color': 'g', 'ls': '-.'}, hist=False, label='Murder')
sns.distplot(df.Assault, bins=30, kde_kws={'color': 'b', 'ls': '-.'}, hist = False, label='Assault')
sns.distplot(df.UrbanPop, bins=30, kde_kws={'color': 'm', 'ls': '-.'}, hist=False, label='UrbanPop')
sns.distplot(df.Rape, bins=30, kde_kws={'color': 'y', 'ls': '-.'}, hist=False, label='Rape')
plt.vlines(x=0, ymin=0.00, ymax=0.40, color='r', linestyles='dotted', alpha=0.4)
sns.distplot(np.random.normal(loc=0,scale=1,size=1000), kde_kws={'color': 'r', 'ls': 'dotted', 'alpha': 0.4}, hist=False, label='Standard Normal Distribution')
plt.xlabel('crime', fontsize=20, color='c')
plt.ylabel('distribution', fontsize=20, color='c')
plt.title('standard normal transformation of crim data', fontsize=30, color='m')
plt.legend()

# COMMAND ----------

corrdf = pd.DataFrame()

for i in df.index.values:
    for j in df.index.values:
        cor = 1 - np.correlate(df.iloc[i], df.iloc[j])
        corrdf = corrdf.append(pd.DataFrame([cor]))

corrdf.columns = ['corr']
corrdf.reset_index(drop=True, inplace=True)
corrdf.head()

# COMMAND ----------

eucldf = pd.DataFrame()

for i in df.index.values:
    for j in df.index.values:
        eucl = (np.linalg.norm(df.iloc[j] - df.iloc[i]))**2
        eucldf = eucldf.append(pd.DataFrame([eucl]))

eucldf.columns = ['eucl']
eucldf.reset_index(drop=True, inplace=True)
eucldf.head()

# COMMAND ----------

maindf = pd.concat([corrdf, eucldf], axis=1)
maindf.head()

# COMMAND ----------

maindf['ratio'] = maindf['corr'] / maindf['eucl']
maindf.head()

# COMMAND ----------

rows_max = maindf.loc[maindf.ratio==maindf.ratio.max()]
rows_max

# COMMAND ----------

maindf.drop(rows_max.index.values, inplace=True)

# COMMAND ----------

maindf.loc[maindf.ratio==maindf.ratio.max()]

# COMMAND ----------

rows_min = maindf.loc[maindf.ratio==maindf.ratio.min()]
rows_min

# COMMAND ----------

maindf.drop(rows_min.index.values, inplace=True)

# COMMAND ----------

maindf.loc[maindf.ratio==maindf.ratio.min()]

# COMMAND ----------

# MAGIC %md
# MAGIC *It is alright to remove ratios with inf of -inf because they denote division by 0. This suggests that the deleted rows essentially are calculating correlations for the same state, which is not required.*

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(maindf.ratio)

# COMMAND ----------

maindf['ratio'].describe().round(2)

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, with an exception of a couple of values, the two measures are similar on an average.

# COMMAND ----------

