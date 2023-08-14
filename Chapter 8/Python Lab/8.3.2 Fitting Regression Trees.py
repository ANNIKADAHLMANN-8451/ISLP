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
from sklearn.tree import DecisionTreeRegressor, plot_tree

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index("SlNo")

Boston = pd.get_dummies(Boston, columns=['chas'], drop_first=True)
Boston.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Regression trees

# COMMAND ----------

X = Boston.drop('medv', axis = 1)
y = Boston.medv

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, test_size = 0.5, random_state = 0)
tree_Boston = DecisionTreeRegressor(max_depth = 2).fit(X_train, y_train)

# COMMAND ----------

plot_tree(tree_Boston)

# COMMAND ----------

pred = tree_Boston.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)

plt.xlabel('pred', color='c', fontsize=20)
plt.ylabel('y_test', color='c', fontsize=20)
plt.title('mean squared error of predicted values', fontsize=30, color='m')

print("Mean Squared Error: ", mean_squared_error(y_test, pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pruning the regression tree

# COMMAND ----------

SCORES = []
max_leafs_arr = range(2, 50)
for max_leafs in max_leafs_arr:
    regressionTree = DecisionTreeRegressor(max_leaf_nodes=max_leafs)
    sc = cross_val_score(regressionTree, X, y, cv=10, scoring="neg_mean_squared_error")
    SCORES.append((-sc.mean(), sc.std()))
SCORES = np.array(SCORES)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(max_leafs_arr, SCORES[:,0], 'g')
plt.fill_between(max_leafs_arr, SCORES[:,0]+SCORES[:,1], SCORES[:,0]-SCORES[:,1], alpha=0.3, color='y')
plt.xlabel('tree size', fontsize=20, color='c')
plt.ylabel('MSE', fontsize=20, color='c')
plt.title('finding the best tree through cross-validation', fontsize=30, color='m')
best_min_leafs = max_leafs_arr[np.argmin(SCORES[:,0])]
print(f"The best tree has {best_min_leafs} leafs.")

# COMMAND ----------

tree_Boston = DecisionTreeRegressor(max_depth = 18).fit(X_train, y_train)

# COMMAND ----------

plot_tree(tree_Boston)

# COMMAND ----------

pred = tree_Boston.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)

plt.xlabel('pred', color='c', fontsize=20)
plt.ylabel('y_test', color='c', fontsize=20)
plt.title('mean squared error of predicted values', fontsize=30, color='m')

print("Mean Squared Error: ", mean_squared_error(y_test, pred))

# COMMAND ----------

# MAGIC %md
# MAGIC **There is a reduction in mean squared error with pruning.**