# Databricks notebook source
# MAGIC %pip install --quiet mlxtend

# COMMAND ----------

# MAGIC %md
# MAGIC In this exercise, we will predict the number of applications received
# MAGIC using the other variables in the `College` data set.

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
from sklearn.metrics import r2_score as r2, mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_linear_regression as PLS

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/College.csv"
College = spark.read.option("header", "true").csv(url).toPandas()
College.set_index('_c0', inplace=True)

str_cols = ["Private"]
float_cols = ["S.F.Ratio"]
int_cols = list(set(College.columns)-set(str_cols)-set(float_cols))
College[int_cols] = College[int_cols].astype(int)
College[str_cols] = College[str_cols].astype(str)
College[float_cols] = College[float_cols].astype(float)

# COMMAND ----------

College.head()

# COMMAND ----------

College.info()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.heatmap(College.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

# COMMAND ----------

# MAGIC %md
# MAGIC *There are no missing values as suggested by the heatmap above.*

# COMMAND ----------

College = pd.get_dummies(College, drop_first=True)

# COMMAND ----------

College.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Split the data set into a training set and a test set.**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a linear model using least squares on the training set, and
# MAGIC report the test error obtained.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Fit a ridge regression model on the training set, with λ chosen
# MAGIC by cross-validation. Report the test error obtained.**

# COMMAND ----------

from sklearn.linear_model import Ridge, RidgeCV

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Fit a lasso model on the training set, with λ chosen by crossvalidation. Report the test error obtained, along with the number of non-zero coefcient estimates.**

# COMMAND ----------

from sklearn.linear_model import Lasso, LassoCV
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Fit a PCR model on the training set, with M chosen by crossvalidation. Report the test error obtained, along with the value
# MAGIC of M selected by cross-validation.**

# COMMAND ----------

from sklearn.preprocessing import scale
from sklearn.model_selection import KFold as KF
from sklearn.decomposition import PCA

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Fit a PLS model on the training set, with M chosen by crossvalidation. Report the test error obtained, along with the value
# MAGIC of M selected by cross-validation.**

# COMMAND ----------

from sklearn.cross_decomposition import PLSRegression as PLS

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Comment on the results obtained. How accurately can we predict the number of college applications received? Is there much
# MAGIC diference among the test errors resulting from these five approaches?**

# COMMAND ----------

# TODO: your response here