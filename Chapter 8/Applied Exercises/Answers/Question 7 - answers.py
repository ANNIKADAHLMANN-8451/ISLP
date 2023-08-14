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

# COMMAND ----------

X = Boston.drop('medv', axis = 1)
y = Boston.medv

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, test_size = 0.5, random_state = 0)

# COMMAND ----------

p = [13, 6, 4]

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.title('MSE for different number of trees', fontsize=30, color='m')
plt.xlabel('number of trees', fontsize=20, color='c')
plt.ylabel('MSE', fontsize=20, color='c')

for j in p:
    MSE = pd.DataFrame()
    for k in range(1,500):
        rf_Boston = RandomForestRegressor(max_features=6, n_estimators=k).fit(X_train, y_train)
        rf_pred = rf_Boston.predict(X_test)
        mse = mean_squared_error(y_test, rf_pred)
        MSE = MSE.append([mse])
        
        MSE.reset_index(drop=True, inplace=True)

        plt.plot(MSE)