# Databricks notebook source
# MAGIC %md
# MAGIC In this problem, you will generate simulated data, and then perform
# MAGIC PCA and K-means clustering on the data.

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
# MAGIC **a. Generate a simulated data set with 20 observations in each of
# MAGIC three classes (i.e. 60 observations total), and 50 variables.**
# MAGIC <br>
# MAGIC <br>
# MAGIC *Hint: There are a number of functions in Python that you can
# MAGIC use to generate data. One example is the `normal()` method of
# MAGIC the `random()` function in numpy; the `uniform()` method is another
# MAGIC option. Be sure to add a mean shift to the observations in each
# MAGIC class so that there are three distinct classes.*

# COMMAND ----------

X = pd.DataFrame()
const = np.random.randint(low=100, size=1)

for num in range(0, 20):
    Z = np.random.normal(loc=0.0, scale=0.1, size=50)
    X = X.append(pd.DataFrame([Z]))
for num in range(0, 20):
    Z = np.random.normal(loc=1.0, scale=0.1, size=50)
    X = X.append(pd.DataFrame([Z]))
for num in range(0, 20):
    Z = np.random.normal(loc=2.0, scale=0.1, size=50)
    X = X.append(pd.DataFrame([Z]))

df = X
cols = np.linspace(0,49, num=50).astype(int)
df.columns = cols

df.head()

# COMMAND ----------

# define class labels
y = pd.DataFrame(index=np.arange(60), columns=np.arange(1))
y.iloc[0:20] = 1
y.iloc[20:40] = 2
y.iloc[40:60] = 3

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Perform PCA on the 60 observations and plot the first two principal component score vectors. Use a diferent color to indicate
# MAGIC the observations in each of the three classes. If the three classes
# MAGIC appear separated in this plot, then continue on to part (c). If
# MAGIC not, then return to part (a) and modify the simulation so that
# MAGIC there is greater separation between the three classes. Do not
# MAGIC continue to part (c) until the three classes show at least some
# MAGIC separation in the frst two principal component score vectors**

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# COMMAND ----------

pca = PCA(n_components=50)
pca_data = pca.fit_transform(df)
principaldf = pd.DataFrame(data = pca_data)
principaldf.head()

# COMMAND ----------

loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df.columns, columns=principaldf.columns)
loadings_df.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
colors = ['red' if yy == 1 else 'green' if yy == 2 else 'blue' for yy in y[0]] # 'red', 'green', 'blue'
plt.scatter(principaldf[0], principaldf[1], s=500, color=colors)

# plt.scatter(principaldf[0], principaldf[1], s=500)
plt.title('principal components', fontsize=30, color='m')
plt.xlabel('principal component 1', fontsize=20, color='c')
plt.ylabel('principal component 2', fontsize=20, color='c')
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 5e-17* np.sqrt(length)
    arrow = plt.arrow(0, 0, pca.mean_[0] + v[0], pca.mean_[1] + v[1], label='mylabel', width=0.005, facecolor='orange', edgecolor='orange', alpha=0.5, )

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Perform K-means clustering of the observations with K = 3.
# MAGIC How well do the clusters that you obtained in K-means clustering compare to the true class labels?**
# MAGIC <br>
# MAGIC <br>
# MAGIC *Hint: You can use the `pd.crosstab()` function in Python to compare the true class labels to the class labels obtained by clustering. Be careful how you interpret the results: K-means clustering
# MAGIC will arbitrarily number the clusters, so you cannot simply check
# MAGIC whether the true class labels and clustering labels are the same.*

# COMMAND ----------

from sklearn.cluster import KMeans as KM
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

km_out = KM(n_clusters=3, n_init=20).fit(df)
km_pred = km_out.predict(df)

# COMMAND ----------

km_labels = pd.DataFrame(km_out.labels_)
km_labels[0].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Perfect match!!!

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Perform K-means clustering with K = 2. Describe your results**

# COMMAND ----------

from sklearn.cluster import KMeans as KM
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

km_out = KM(n_clusters=2, n_init=20).fit(df)
km_pred = km_out.predict(df)

# COMMAND ----------

km_labels = pd.DataFrame(km_out.labels_)
km_labels[0].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC All of values in `class=2` gets transferred into `class=0`.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now perform K-means clustering with K = 4, and describe your
# MAGIC results.**

# COMMAND ----------

from sklearn.cluster import KMeans as KM
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

km_out = KM(n_clusters=4, n_init=20).fit(df)
km_pred = km_out.predict(df)

# COMMAND ----------

km_labels = pd.DataFrame(km_out.labels_)
km_labels[0].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC A new `class=3` is created and the values in `class=2` is distributed between these two classes.

# COMMAND ----------

# MAGIC %md
# MAGIC **f.  Now perform K-means clustering with K = 3 on the frst two
# MAGIC principal component score vectors, rather than on the raw data.
# MAGIC That is, perform K-means clustering on the 60 × 2 matrix of
# MAGIC which the frst column is the frst principal component score
# MAGIC vector, and the second column is the second principal component
# MAGIC score vector. Comment on the results.**

# COMMAND ----------

pr_df = principaldf[[0,1]]
pr_df.head()

# COMMAND ----------

km_out = KM(n_clusters=3, n_init=20).fit(pr_df)
km_pred = km_out.predict(pr_df)

# COMMAND ----------

km_labels = pd.DataFrame(km_out.labels_)
km_labels[0].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Perfect match!!!

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Using the `StandardScaler()` estimator, perform K-means clustering with K = 3 on the data after scaling each variable to have
# MAGIC standard deviation one. How do these results compare to those
# MAGIC obtained in (b)? Explain.**

# COMMAND ----------

df = pd.DataFrame(scale(df))

# COMMAND ----------

df.describe().round(2)

# COMMAND ----------

km_out = KM(n_clusters=3, n_init=20).fit(df)
km_pred = km_out.predict(df)

# COMMAND ----------

km_labels = pd.DataFrame(km_out.labels_)
km_labels[0].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Some misclassification, which means poorer results that performing KMeans without scaling with `K=3` as well as performing KMeans on principal components with `K=3`, but better than rest.