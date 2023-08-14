# Databricks notebook source
# MAGIC %pip install --quiet mlxtend

# COMMAND ----------

# MAGIC %md
# MAGIC We will now try to predict per capita crime rate in the `Boston` data
# MAGIC set. Note, this notebook was modified from the original repo as the `normalize` parameter in some of the model instance functions has since been removed.

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
from sklearn.preprocessing import scale

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index("SlNo", inplace=True)

# COMMAND ----------

Boston.head()

# COMMAND ----------

Boston = pd.get_dummies(Boston, columns =['chas'], drop_first=True)

# COMMAND ----------

X = Boston.drop(columns='crim')
y = Boston['crim']

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Try out some of the regression methods explored in this chapter,
# MAGIC such as best subset selection, the lasso, ridge regression, and
# MAGIC PCR. Present and discuss results for the approaches that you
# MAGIC consider.**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC Lasso

# COMMAND ----------

from sklearn.linear_model import Lasso, LassoCV
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC Ridge regression

# COMMAND ----------

from sklearn.linear_model import Ridge, RidgeCV

# COMMAND ----------

# MAGIC %md
# MAGIC Principal components regression

# COMMAND ----------

from sklearn.preprocessing import scale
from sklearn.model_selection import KFold as KF, cross_val_score
from sklearn.decomposition import PCA

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Propose a model (or set of models) that seem to perform well on
# MAGIC this data set, and justify your answer. Make sure that you are
# MAGIC evaluating model performance using validation set error, crossvalidation, or some other reasonable alternative, as opposed to
# MAGIC using training error.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Does your chosen model involve all of the features in the data
# MAGIC set? Why or why not?**

# COMMAND ----------

# TODO: your response here