# Databricks notebook source
# MAGIC %pip install mlxtend

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
from sklearn.metrics import r2_score as r2
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_linear_regression as PLS

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

# clean data
print(Hitters.shape)
Hitters = Hitters.dropna()

# COMMAND ----------

Hitters.shape

# COMMAND ----------

Hitters.head()

# COMMAND ----------

# converting categorical data into dummy variable
Hitters_1 = pd.get_dummies(Hitters, drop_first=True, columns=['League', 'Division', 'NewLeague'])

# COMMAND ----------

Hitters_1.head()

# COMMAND ----------

# best subset selection
X = Hitters_1.drop(columns = ['Salary', 'Names'])
y = Hitters_1.Salary

# COMMAND ----------

lm = LinearRegression()
efs = EFS(lm, min_features=1, max_features=8, scoring='r2', cv=3).fit(X,y)

# COMMAND ----------

features = efs.best_feature_names_
features

# COMMAND ----------

idx = efs.best_idx_
idx

# COMMAND ----------

feat_list = pd.concat([pd.DataFrame([features]).T, pd.DataFrame([idx]).T], axis = 1)
feat_list.columns = ['Features', 'Index']
avg_score = pd.DataFrame()
for i in idx:
    avg_score = avg_score.append(pd.DataFrame([efs.subsets_[i]['avg_score']]))
avg_score.columns = ['Average Score']
avg_score.reset_index(drop=True,inplace=True)
feat_list = pd.concat([feat_list, avg_score], axis=1, ignore_index=True)
feat_list.columns = ['Features', 'Index', 'Average Score']
feat_list

# COMMAND ----------

X_reg = X[feat_list['Features']]

# COMMAND ----------

X_reg.head()

# COMMAND ----------

lm_best = LinearRegression().fit(X_reg, y)

# COMMAND ----------

final_coefs = pd.DataFrame(lm_best.coef_)
final_coefs = pd.concat([feat_list['Features'], final_coefs], axis = 1)
final_intercept = pd.DataFrame([lm_best.intercept_]*8)
final_coefs = pd.concat([final_coefs, final_intercept], axis = 1)
final_coefs.columns = ['Features', 'Coefficients', 'Intercept']
final_coefs

# COMMAND ----------

Xj = pd.DataFrame()
r2iPlot = pd.DataFrame()
for i in final_coefs['Features']:
    Xj = pd.concat([Xj, X[[i]]], axis = 1)
    lmi = LinearRegression().fit(Xj, y)
    lmpred = lmi.predict(Xj)
    r2i = r2(y, lmpred)
    r2iPlot = r2iPlot.append(pd.DataFrame([r2i]))
r2iPlot.columns = ['R2']
r2iPlot.reset_index(inplace=True, drop=True)
r2iPlot.index = r2iPlot.index + 1
r2iPlot

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(r2iPlot, marker='o', color='green')
plt.title('Best subset selection')
plt.ylabel('R^2')
plt.xlabel('Number of features')
plt.grid()

# COMMAND ----------

# MAGIC %md
# MAGIC **As we can see the model gets its higest $R^2$ when the number of features n=8.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### A BIG downside to best subset selection is that it's computationally intensive and takes a lot of time. In this particular instance for example, it took me a little less than an hour to find the best subsets!