# Databricks notebook source
# MAGIC %pip install mlxtend

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# import data and preprocess
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index("SlNo")

Boston = pd.get_dummies(Boston, columns=['chas'], drop_first=True)
Boston.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bagging

# COMMAND ----------

X = Boston.drop('medv', axis = 1)
y = Boston.medv

# COMMAND ----------

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, test_size = 0.5, random_state = 0)
bag_Boston = RandomForestRegressor(max_features=13).fit(X_train, y_train)

bag_pred = bag_Boston.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(bag_pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print("Mean Squared Error: ", mean_squared_error(y_test, bag_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC **As you can see, bagging already provides significant reduction in mean squared error over regression models.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

rf_Boston = RandomForestRegressor(max_features=6).fit(X_train, y_train)

rf_pred = rf_Boston.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(rf_pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print("Mean Squared Error: ", mean_squared_error(y_test, rf_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC **Random forests yield an improvement over bagging.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Viewing importance of each variable

# COMMAND ----------

Importance = pd.DataFrame({'Importance': rf_Boston.feature_importances_*100}, index = X.columns)
Importance.sort_values(by = 'Importance', axis = 0, ascending = True).plot(kind = 'barh', color = 'y')

plt.xkcd()
plt.xlabel('variable importance', fontsize=20, color='c')
plt.ylabel('variables', fontsize=20, color='c')
plt.title('importance of each variables', fontsize=30, color='m')
plt.gca().legend_ = None
fig = plt.gcf()
fig.set_size_inches(25, 10)

# COMMAND ----------

