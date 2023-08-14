# Databricks notebook source
# MAGIC %md
# MAGIC In Section 12.2.3, a formula for calculating PVE was given in Equation 12.10. We also saw that the PVE can be obtained using the
# MAGIC `explained_variance_ratio_` attribute of a fitted `PCA()` estimator. On the `USArrests` data, calculate PVE in the two ways highlighted below.

# COMMAND ----------

# MAGIC %md
# MAGIC These two approaches should give the same results.
# MAGIC <br>
# MAGIC <br>
# MAGIC *Hint: You will only obtain the same results in (a) and (b) if the same data is used in both cases. For instance, if in (a) you performed PCA() using centered and scaled variables, then you must center and scale the variables before applying Equation 12.10 in (b).*

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

df = pd.DataFrame(scale(USArrests))
df.columns = USArrests.columns
df.index = USArrests.index
df.head()

# COMMAND ----------

df.describe().round(4)

# COMMAND ----------

pca = PCA(n_components=4)
pca_data = pca.fit_transform(df)
principaldf = pd.DataFrame(data = pca_data, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
principaldf.head()

# COMMAND ----------

principaldf.info()

# COMMAND ----------

PVAR = principaldf.var()
PVAR

# COMMAND ----------

PSUM = np.sum(PVAR)
PSUM

# COMMAND ----------

PVE_method = pd.DataFrame([PVAR/PSUM]).T
PVE_method.columns = ['explained variance ratio']
PVE_method.index = principaldf.columns
PVE_method

# COMMAND ----------

loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df.columns, columns=principaldf.columns)
loadings_df

# COMMAND ----------

# MAGIC %md
# MAGIC **b. By applying Equation 12.10 directly. The loadings are stored
# MAGIC as the `components_` attribute of the fitted `PCA()` estimator. Use
# MAGIC those loadings in Equation 12.10 to obtain the PVE.**

# COMMAND ----------

# PC1
num = np.sum((np.dot(df, loadings_df.PC1))**2)

denomdf = pd.DataFrame()
for i in range(0, 50):
    row_sum = np.sum(df.iloc[i]**2)
    denomdf = denomdf.append(pd.DataFrame([row_sum]))

denomdf.columns = ['sums']
denomdf.reset_index(drop=True, inplace=True)
denom = denomdf.sum()

PVE_PC1 = num/denom
PVE_PC1

# COMMAND ----------

# PC2
num = np.sum((np.dot(df, loadings_df.PC2))**2)

denomdf = pd.DataFrame()
for i in range(0, 50):
    row_sum = np.sum(df.iloc[i]**2)
    denomdf = denomdf.append(pd.DataFrame([row_sum]))

denomdf.columns = ['sums']
denomdf.reset_index(drop=True, inplace=True)
denom = denomdf.sum()

PVE_PC2 = num/denom
PVE_PC2

# COMMAND ----------

# PC3
num = np.sum((np.dot(df, loadings_df.PC3))**2)

denomdf = pd.DataFrame()
for i in range(0, 50):
    row_sum = np.sum(df.iloc[i]**2)
    denomdf = denomdf.append(pd.DataFrame([row_sum]))

denomdf.columns = ['sums']
denomdf.reset_index(drop=True, inplace=True)
denom = denomdf.sum()

PVE_PC3 = num/denom
PVE_PC3

# COMMAND ----------

# PC4
num = np.sum((np.dot(df, loadings_df.PC4))**2)

denomdf = pd.DataFrame()
for i in range(0, 50):
    row_sum = np.sum(df.iloc[i]**2)
    denomdf = denomdf.append(pd.DataFrame([row_sum]))

denomdf.columns = ['sums']
denomdf.reset_index(drop=True, inplace=True)
denom = denomdf.sum()

PVE_PC4 = num/denom
PVE_PC4

# COMMAND ----------

PVE_formula = pd.DataFrame([PVE_PC1.values, PVE_PC2.values, PVE_PC3.values, PVE_PC4.values])
PVE_formula.columns = ['explained variance ratio']
PVE_formula.index = principaldf.columns
PVE_formula

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, PVE through both method and formula are the same.

# COMMAND ----------

