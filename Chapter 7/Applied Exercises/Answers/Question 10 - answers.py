# Databricks notebook source
# MAGIC %md
# MAGIC This question relates to the `College` data set.

# COMMAND ----------

# MAGIC %pip install --quiet mlxtend

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd

# COMMAND ----------

# import relevant data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import custom packages
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_linear_regression as PLS
from sklearn.preprocessing import scale
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/College.csv"
College = spark.read.option("header", "true").csv(url).toPandas()
College.set_index("_c0", inplace=True)

str_cols = ["Private"]
float_cols = ["S.F.Ratio"]
int_cols = list(set(College.columns)-set(str_cols)-set(float_cols))
College[str_cols] = College[str_cols].astype(str)
College[float_cols] = College[float_cols].astype(float)
College[int_cols] = College[int_cols].astype(int)

# COMMAND ----------

College.head()

# COMMAND ----------

College.info()

# COMMAND ----------

College = pd.get_dummies(data=College, columns=['Private'], drop_first=True)

# COMMAND ----------

College.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Split the data into a training set and a test set. Using out-of-state
# MAGIC tuition as the response and the other variables as the predictors,
# MAGIC perform forward stepwise selection on the training set in order
# MAGIC to identify a satisfactory model that uses just a subset of the
# MAGIC predictors.**

# COMMAND ----------

from sklearn.model_selection import train_test_split
from pygam import LinearGAM
from pygam.terms import gen_edge_knots
from sklearn.ensemble.partial_dependence import plot_partial_dependence

# COMMAND ----------

X = College.drop(columns='Outstate')
y = College.Outstate

# COMMAND ----------

lmf = LinearRegression()
sfs = SFS(lmf, k_features=(1,len(X.columns)), forward=True, floating=False, scoring='r2',cv=10)

# COMMAND ----------

plt.xkcd()
sfs = sfs.fit(X.values, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err', color='green')
plt.title('Forward Stepwise Selection')
plt.ylabel('R^2')
plt.grid()
fig = plt.gcf()
fig.set_size_inches(25, 10)
[plt.hlines(0.735 , 0, 18, linestyles='dashed', lw=2, colors='c')]
[plt.vlines(12 , 0, 0.8, linestyles='dashed', lw=2, colors='c')]

# COMMAND ----------

# MAGIC %md
# MAGIC Forward stepwise selection achieves the highest $R^2$ for n=12. This means that the best regression equation will contain 12 features when employing forward stepwise selection.

# COMMAND ----------

feat = sfs.k_feature_idx_
feat_list = pd.DataFrame()
for i in feat:
    print(X.columns[i])
    feat_list = feat_list.append(pd.DataFrame([X.columns[i]]))
feat_list.columns = ['Features']
feat_list.reset_index(inplace=True, drop=True)
feat_list.head()