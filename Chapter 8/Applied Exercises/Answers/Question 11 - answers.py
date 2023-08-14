# Databricks notebook source
# MAGIC %md
# MAGIC This question uses the `Caravan` data set.

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

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Caravan.csv"
Caravan = spark.read.option("header", "true").csv(url).toPandas()
Caravan.set_index('_c0', inplace=True)
Caravan.index.names = ['Index']

str_cols = ["Purchase"]
num_cols = list(set(Caravan.columns) - set(str_cols))
Caravan[str_cols] = Caravan[str_cols].astype(str)
Caravan[num_cols] = Caravan[num_cols].astype(float)

# COMMAND ----------

Caravan.Purchase.value_counts()

# COMMAND ----------

Caravan.Purchase = Caravan.Purchase.map({'No': 0, 'Yes': 1})
Caravan.head()

# COMMAND ----------

Caravan.Purchase.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a training set consisting of the frst 1,000 observations,
# MAGIC and a test set consisting of the remaining observations.**

# COMMAND ----------

X = Caravan.drop(columns='Purchase')
y = Caravan.Purchase

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.82823771899, random_state=42)

# COMMAND ----------

len(X_train), len(y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC **b.  Fit a boosting model to the training set with `Purchase` as the
# MAGIC response and the other variables as predictors. Use 1,000 trees,
# MAGIC and a shrinkage value of 0.01. Which predictors appear to be
# MAGIC the most important?**

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

# MAGIC %md
# MAGIC 'PPERSAUT', 'MOSTYPE', 'MFGEKIND', 'MBERABO' and 'MINK3045' are the most important predictors. The meaning of these variables can be found at: http://bit.ly/2Bb2N2J

# COMMAND ----------

print("Mean Squared Error: ", mean_squared_error(y_test, boost.predict(X_test)))

# COMMAND ----------

boost_pred = pd.DataFrame(boost.predict(X_test))
boost_pred.columns = ['Predictions']
boost_pred.head()

# COMMAND ----------

boost_prob = np.where(boost_pred.Predictions>0.2, 1, 0)
boost_prob

# COMMAND ----------

class_mat = pd.DataFrame(confusion_matrix(y_test, boost_prob).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
class_mat

# COMMAND ----------

print(classification_report(y_test, boost_prob))

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, it correctly predicts for $\approx$86% of the test data set.

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Use the boosting model to predict the response on the test data.
# MAGIC Predict that a person will make a purchase if the estimated probability of purchase is greater than 20 %. Form a confusion matrix. What fraction of the people predicted to make a purchase
# MAGIC do in fact make one? How does this compare with the results
# MAGIC obtained from applying KNN or logistic regression to this data
# MAGIC set?**

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

lmreg = LinearRegression().fit(X_train, y_train)
lmpred = pd.DataFrame(lmreg.predict(X_test))
lmpred.columns = ['Predictions']

# COMMAND ----------

lmpred.Predictions = np.where(lmpred.Predictions>0.2, 1, 0)
class_mat = pd.DataFrame(confusion_matrix(y_test, lmpred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
class_mat

# COMMAND ----------

print(classification_report(y_test, lmpred))

# COMMAND ----------

# MAGIC %md
# MAGIC Linear regression provides better results to those of boosting.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

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
# MAGIC Logistic regression provides even better predictions than linear regression and boosting.