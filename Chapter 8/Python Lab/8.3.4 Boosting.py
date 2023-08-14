# Databricks notebook source
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index("SlNo")

Boston = pd.get_dummies(Boston, columns=['chas'], drop_first=True)
Boston.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Boosting ($\lambda$=0.001)

# COMMAND ----------

X = Boston.drop('medv', axis = 1)
y = Boston.medv

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, test_size = 0.5, random_state = 0)
bag_Boston = RandomForestRegressor(max_features=13).fit(X_train, y_train)

# COMMAND ----------

boost_Boston = GradientBoostingRegressor(n_estimators=5000, max_depth=4, learning_rate=0.001).fit(X_train, y_train)
print("Mean Squared Error: ", mean_squared_error(y_test, boost_Boston.predict(X_test)))

# COMMAND ----------

feature_importance = boost_Boston.feature_importances_*100
rel_imp = pd.Series(feature_importance, index = X.columns).sort_values(inplace = False)

rel_imp.T.plot(kind = 'barh', color = 'y', figsize=(25, 10), grid= True, )

plt.xkcd()
plt.xlabel('variable importance', fontsize=20, color='c')
plt.ylabel('variables', fontsize=20, color='c')
plt.title('importance of each variables', fontsize=30, color='m')
plt.gca().legend_ = None

# COMMAND ----------

# MAGIC %md
# MAGIC **From the above diagramme, it is clear that $lstat$ and $rm$ are the most important variables. What is important to note is that the MSE of boosting is similar to that of bagging. Therefore, I will need to use a different value of the tuning parameter, $\lambda$.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Boosting ($\lambda$=0.2)

# COMMAND ----------

boost_Boston = GradientBoostingRegressor(n_estimators=5000, max_depth=4, learning_rate=0.2).fit(X_train, y_train)
print("Mean Squared Error: ", mean_squared_error(y_test, boost_Boston.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Therefore using $\lambda$=0.2 leads to a slightly lower test MSE than $\lambda$=0.0001.**