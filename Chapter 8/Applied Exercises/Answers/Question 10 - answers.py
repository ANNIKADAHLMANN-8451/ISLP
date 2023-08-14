# Databricks notebook source
# MAGIC %md
# MAGIC We now use boosting to predict `Salary` in the `Hitters` data set.

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
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Hitters.csv"
Hitters = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["Names", "NewLeague", "League", "Division"]
num_cols = list(set(Hitters.columns) - set(str_cols))
Hitters["Salary"] = np.where(Hitters["Salary"] == "NA", np.nan, Hitters["Salary"])
Hitters[str_cols] = Hitters[str_cols].astype(str)
Hitters[num_cols] = Hitters[num_cols].astype(float)

# COMMAND ----------

Hitters.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Remove the observations for whom the salary information is
# MAGIC unknown, and then log-transform the salaries.**

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(Hitters.isna(), cmap='viridis', yticklabels=False, cbar=False)
plt.title('heatmap to visualise missing data', fontsize=30, color='m')
plt.xlabel('features', fontsize=20, color='c')

# COMMAND ----------

Hitters.dropna(axis=0, inplace=True)

# COMMAND ----------

Hitters.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(Hitters.isna(), cmap='viridis', yticklabels=False, cbar=False)
plt.title('heatmap to visualise missing data', fontsize=30, color='m')
plt.xlabel('features', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC So, I have removed all observations where Salary information is unknown.

# COMMAND ----------

Hitters.Salary = np.log(Hitters.Salary)

# COMMAND ----------

Hitters.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, I have log-transformed the salaries.

# COMMAND ----------

Hitters.League.value_counts()

# COMMAND ----------

Hitters.Division.value_counts()

# COMMAND ----------

Hitters.NewLeague.value_counts()

# COMMAND ----------

Hitters.League = Hitters.League.map({'N': 0, 'A': 1})
Hitters.Division = Hitters.Division.map({'W': 0, 'E': 1})
Hitters.NewLeague = Hitters.NewLeague.map({'N': 0, 'A': 1})

# COMMAND ----------

Hitters.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Create a training set consisting of the frst 200 observations, and
# MAGIC a test set consisting of the remaining observations.**

# COMMAND ----------

X = Hitters.drop(columns=['Salary', 'Names'])
y = Hitters.Salary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23954372623, random_state=42)

# COMMAND ----------

X_train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Perform boosting on the training set with 1,000 trees for a range
# MAGIC of values of the shrinkage parameter λ. Produce a plot with
# MAGIC diferent shrinkage values on the x-axis and the corresponding
# MAGIC training set MSE on the y-axis.**

# COMMAND ----------

SP = np.linspace(start=0.001, stop=0.9, num=100)
MSE = pd.DataFrame()

for k in SP:
    boost = GradientBoostingRegressor(n_estimators=1000, max_depth=4, learning_rate=k).fit(X_train, y_train)
    mse = mean_squared_error(y_test, boost.predict(X_test))
    MSE = MSE.append([mse])

MSE.columns = ['MSE']
MSE.reset_index(drop=True, inplace=True)

# COMMAND ----------

MSE.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Produce a plot with diferent shrinkage values on the x-axis and
# MAGIC the corresponding test set MSE on the y-axis.**

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(MSE, SP, alpha=1)
sns.regplot(MSE, SP, x_ci='0.95', line_kws={'color': 'g', 'ls': '-.'})
plt.title('MSE vs shrinkage values', fontsize=30, color='m')
plt.xlabel('MSE', fontsize=20, color='c')
plt.ylabel('shrinkage values', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Compare the test MSE of boosting to the test MSE that results
# MAGIC from applying two of the regression approaches seen in
# MAGIC Chapters 3 and 6.**

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

lmreg = LinearRegression().fit(X_train, y_train)
lmpred = lmreg.predict(X_test)
print("MSE from linear regression: ", mean_squared_error(y_test, lmpred))

# COMMAND ----------

print("MSE from boosting: ", MSE.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, boosting provides lower MSE than linear regression.

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Which variables appear to be the most important predictors in
# MAGIC the boosted model?**

# COMMAND ----------

feature_importance = boost.feature_importances_*100
rel_imp = pd.Series(feature_importance, index = X.columns).sort_values(inplace = False)

rel_imp.T.plot(kind = 'barh', color = 'y', figsize=(25, 10), grid= True, )

plt.xkcd()
plt.xlabel('variable importance', fontsize=20, color='c')
plt.ylabel('variables', fontsize=20, color='c')
plt.title('importance of each variables', fontsize=30, color='m')
plt.gca().legend_ = None

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, 'CRuns', 'CRBI' and 'AtBat' are the most important variables.

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Now apply bagging to the training set. What is the test set MSE
# MAGIC for this approach?**

# COMMAND ----------

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

bag = RandomForestRegressor(max_features=19).fit(X_train, y_train)
bag_pred = bag.predict(X_test)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(bag_pred, y_test, label = 'medv', color='g')
plt.plot([0, 1], [0, 1], 'r', transform = plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

print("Mean Squared Error: ", mean_squared_error(y_test, bag_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC The test MSE for bagging is $\approx$0.25, which is lower than the test MSE for boosting.

# COMMAND ----------

