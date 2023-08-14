# Databricks notebook source
# MAGIC %md
# MAGIC In Section 8.3.3, we applied random forests to the `Boston` data using
# MAGIC `max_features = 6` and using `n_estimators = 100` and `n_estimators =
# MAGIC 500`. Create a plot displaying the test error resulting from random
# MAGIC forests on this data set for a more comprehensive range of values
# MAGIC for `max_features` and `n_estimators`. You can model your plot after
# MAGIC Figure 8.10. Describe the results obtained.

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# import data
url = "/Users/arpanganguli/Documents/Professional/Finance/ISLR/Boston.csv"
Boston = pd.read_csv(url, index_col='SlNo')
Boston = pd.get_dummies(Boston, columns=['chas'], drop_first=True)
Boston.head()