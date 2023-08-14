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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Perform PCA on the 60 observations and plot the frst two principal component score vectors. Use a diferent color to indicate
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

# TODO: your response here

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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Perform K-means clustering with K = 2. Describe your results**

# COMMAND ----------

from sklearn.cluster import KMeans as KM
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now perform K-means clustering with K = 4, and describe your
# MAGIC results.**

# COMMAND ----------

from sklearn.cluster import KMeans as KM
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f.  Now perform K-means clustering with K = 3 on the frst two
# MAGIC principal component score vectors, rather than on the raw data.
# MAGIC That is, perform K-means clustering on the 60 × 2 matrix of
# MAGIC which the frst column is the frst principal component score
# MAGIC vector, and the second column is the second principal component
# MAGIC score vector. Comment on the results.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Using the `StandardScaler()` estimator, perform K-means clustering with K = 3 on the data after scaling each variable to have
# MAGIC standard deviation one. How do these results compare to those
# MAGIC obtained in (b)? Explain.**

# COMMAND ----------

# TODO: your response here