# Databricks notebook source
# MAGIC %md
# MAGIC  In the lab, a classifcation tree was applied to the `Carseats` data set after converting `Sales` into a qualitative response variable. Now we will
# MAGIC seek to predict `Sales` using regression trees and related approaches,
# MAGIC treating the response as a quantitative variable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

# COMMAND ----------

# import data visualisation packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Carseats.csv"
df = spark.read.option("header", "true").csv(url).toPandas()
df.set_index("SlNo", inplace=True)

str_cols = ["ShelveLoc", "Urban", "US"]
float_cols = ["Sales"]
int_cols = list(set(df.columns)-set(str_cols)-set(float_cols))
df[str_cols] = df[str_cols].astype(str)
df[float_cols] = df[float_cols].astype(float)
df[int_cols] = df[int_cols].astype(int)


df.Sales = df.Sales.map(lambda x: 0 if x<=8 else 1)
df.ShelveLoc = pd.factorize(df.ShelveLoc)[0]
df.Urban = df.Urban.map({'No':0, 'Yes':1})
df.US = df.US.map({'No':0, 'Yes':1})

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Split the data set into a training set and a test set.**

# COMMAND ----------

X = df.drop(['Sales'], axis = 1)
y = df.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, test_size = 0.3, random_state = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a regression tree to the training set. Plot the tree, and interpret the results. What test MSE do you obtain?**

# COMMAND ----------

tree_cs = DecisionTreeRegressor(max_depth = 2).fit(X_train, y_train)

plot_tree(tree_cs)

# COMMAND ----------

pred = tree_cs.predict(X_test)

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
# MAGIC Therefore, the test MSE$\approx$0.22

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Use cross-validation in order to determine the optimal level of
# MAGIC tree complexity. Does pruning the tree improve the test MSE?**

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

tree_cs = DecisionTreeRegressor(max_depth = 4).fit(X_train, y_train)

plot_tree(tree_cs)

# COMMAND ----------

pred = tree_cs.predict(X_test)

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
# MAGIC Pruning in this instance actually *increases* the test MSE.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Use the bagging approach in order to analyze this data. What
# MAGIC test MSE do you obtain? Use the `feature\_importance\_ values` to
# MAGIC determine which variables are most important.**

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

bag_cs = RandomForestRegressor(max_features=10).fit(X_train, y_train)
bag_pred = bag_cs.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(bag_pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print("Mean Squared Error: ", mean_squared_error(y_test, bag_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC Bagging provides significant reduction in mean squared error over regression models.

# COMMAND ----------

Importance = pd.DataFrame({'Importance': bag_cs.feature_importances_*100}, index = X.columns)
Importance.sort_values(by = 'Importance', axis = 0, ascending = True).plot(kind = 'barh', color = 'y')

plt.xkcd()
plt.xlabel('variable importance', fontsize=20, color='c')
plt.ylabel('variables', fontsize=20, color='c')
plt.title('importance of each variables', fontsize=30, color='m')
plt.gca().legend_ = None
fig = plt.gcf()
fig.set_size_inches(25, 10)

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Use random forests to analyze this data. What test MSE do
# MAGIC you obtain? Use the `feature\_importance\_ values` to determine
# MAGIC which variables are most important. Describe the efect of m, the
# MAGIC number of variables considered at each split, on the error rate
# MAGIC obtained.**

# COMMAND ----------

# MAGIC %md
# MAGIC $m$=4

# COMMAND ----------

rf_cs = RandomForestRegressor(max_features=3).fit(X_train, y_train)

rf_pred = rf_cs.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(rf_pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print("Mean Squared Error: ", mean_squared_error(y_test, rf_pred))

# COMMAND ----------

Importance = pd.DataFrame({'Importance': rf_cs.feature_importances_*100}, index = X.columns)
Importance.sort_values(by = 'Importance', axis = 0, ascending = True).plot(kind = 'barh', color = 'y')

plt.xkcd()
plt.xlabel('variable importance', fontsize=20, color='c')
plt.ylabel('variables', fontsize=20, color='c')
plt.title('importance of each variables', fontsize=30, color='m')
plt.gca().legend_ = None
fig = plt.gcf()
fig.set_size_inches(25, 10)

# COMMAND ----------

# MAGIC %md
# MAGIC Random forest decreases the test MSE.

# COMMAND ----------

MSE = pd.DataFrame()

for k in range(1,11):
    rf_cs = RandomForestRegressor(max_features=k).fit(X_train, y_train)
    rf_pred = rf_cs.predict(X_test)
    mse = mean_squared_error(y_test, rf_pred)
    MSE = MSE.append([mse])
    
MSE.reset_index(drop=True, inplace=True)
plt.figure(figsize=(25, 10))
plt.plot(MSE, color='g', ls='-.', marker='o', markersize=10, markerfacecolor='orange')
plt.xlabel('number of variables', color='c', fontsize=20)
plt.ylabel('MSE', color='c', fontsize=20)
plt.title('effect of number of variables on MSE', color='m', fontsize=30)

# COMMAND ----------

# MAGIC %md
# MAGIC Changing m varies test MSE between approx. 0.15 to approx. 0.20.