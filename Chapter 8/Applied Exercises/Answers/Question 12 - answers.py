# Databricks notebook source
# MAGIC %md
# MAGIC Apply boosting, bagging, random forests, and BART to a data set
# MAGIC of your choice. Be sure to ft the models on a training set and to
# MAGIC evaluate their performance on a test set. How accurate are the results
# MAGIC compared to simple methods like linear or logistic regression? Which
# MAGIC of these approaches yields the best performance?

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
from sklearn.metrics import r2_score as r2, mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_linear_regression as PLS
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Heart.csv"
df = spark.read.option("header", "true").csv(url).toPandas()
df.set_index("_c0", inplace=True)

str_cols = ["ChestPain", "Thal", "AHD"]
float_cols = ["Ca", "Oldpeak"]
int_cols = list(set(df.columns)-set(str_cols)-set(float_cols))
df[str_cols] = df[str_cols].astype(str)
df[float_cols] = df[float_cols].astype(float)
df[int_cols] = df[int_cols].astype(int)

# COMMAND ----------

df.ChestPain.value_counts()

# COMMAND ----------

df.Thal.value_counts()

# COMMAND ----------

df.AHD.value_counts()

# COMMAND ----------

df.AHD = df.AHD.map({'No': 0, 'Yes': 1})

# COMMAND ----------

df.Thal = pd.factorize(df.Thal)[0]

# COMMAND ----------

df.ChestPain = pd.factorize(df.ChestPain)[0]

# COMMAND ----------

df.head(25)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('missing values in df', fontsize=30, color='m')
plt.xlabel('features', fontsize=20, color='c')

# COMMAND ----------

df.drop(axis=0, inplace=True, columns='Ca')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('missing values in df', fontsize=30, color='m')
plt.xlabel('features', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we have no missing values in df.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

X = df.drop(columns='AHD')
y = df.AHD

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# COMMAND ----------

logreg = LogisticRegression(solver='liblinear').fit(X_train, y_train)
logpred = pd.DataFrame(logreg.predict(X_test))
logpred.columns = ['Predictions']

# COMMAND ----------

logpred.Predictions = np.where(logpred.Predictions>0.2, 1, 0)
class_mat = pd.DataFrame(confusion_matrix(y_test, logpred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
class_mat

# COMMAND ----------

print(classification_report(y_test, logpred))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Boosting

# COMMAND ----------

boost = GradientBoostingRegressor(n_estimators=1000, max_depth=6, learning_rate=0.01).fit(X_train, y_train)

# COMMAND ----------

feature_importance = boost.feature_importances_*100
rel_imp = pd.Series(feature_importance, index = X.columns).sort_values(inplace = False)

rel_imp.T.plot(kind = 'barh', color = 'y', figsize=(25, 25), grid= True, )

plt.xkcd()
plt.xlabel('variable importance', fontsize=20, color='c')
plt.ylabel('variables', fontsize=20, color='c')
plt.title('importance of each variables', fontsize=30, color='m')
plt.gca().legend_ = None

# COMMAND ----------

print("Mean Squared Error: ", mean_squared_error(y_test, boost.predict(X_test)))

# COMMAND ----------

boost_pred = pd.DataFrame(boost.predict(X_test))
boost_pred.columns = ['Predictions']
boost_pred.head()

# COMMAND ----------

boost_pred.mean()

# COMMAND ----------

boost_prob = np.where(boost_pred.Predictions>0.452202, 1, 0)
boost_prob

# COMMAND ----------

class_mat = pd.DataFrame(confusion_matrix(y_test, boost_prob).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
class_mat

# COMMAND ----------

print(classification_report(y_test, boost_prob))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bagging

# COMMAND ----------

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

bag = RandomForestRegressor(max_features=12).fit(X_train, y_train)
bag_pred = bag.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(bag_pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print("Mean Squared Error: ", mean_squared_error(y_test, bag_pred))

# COMMAND ----------

bag_pred.mean()

# COMMAND ----------

bag_prob = np.where(bag_pred>0.49300000000000005, 1, 0)
bag_prob

# COMMAND ----------

class_mat = pd.DataFrame(confusion_matrix(y_test, bag_prob).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
class_mat

# COMMAND ----------

print(classification_report(y_test, bag_prob))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random forests

# COMMAND ----------

rf = RandomForestRegressor(max_features=5).fit(X_train, y_train)

rf_pred = rf.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(rf_pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print("Mean Squared Error: ", mean_squared_error(y_test, rf_pred))

# COMMAND ----------

rf_pred.mean()

# COMMAND ----------

rf_prob = np.where(bag_pred>0.47699999999999987, 1, 0)
rf_prob

# COMMAND ----------

class_mat = pd.DataFrame(confusion_matrix(y_test, rf_prob).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
class_mat

# COMMAND ----------

print(classification_report(y_test, rf_prob))

# COMMAND ----------

# MAGIC %md
# MAGIC For the Heart data set, logistic regression provides the best results.

# COMMAND ----------

