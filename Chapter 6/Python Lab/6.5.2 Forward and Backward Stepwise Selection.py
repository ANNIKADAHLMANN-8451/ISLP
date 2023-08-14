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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as r2
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

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

# MAGIC %md
# MAGIC ## Forward Stepwise Selection

# COMMAND ----------

lmf = LinearRegression()

# COMMAND ----------

X = Hitters_1.drop(columns = ['Salary', 'Names'])
y = Hitters_1.Salary

# COMMAND ----------

X.head()

# COMMAND ----------

sfs = SFS(lmf, k_features=(1,len(X.columns)), forward=True, floating=False, scoring='r2',cv=10)

# COMMAND ----------

plt.xkcd()
sfs = sfs.fit(X.values, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err', color='green')
plt.title('Forward Stepwise Selection')
plt.ylabel('R^2')
plt.grid()
fig = plt.gcf()
fig.set_size_inches(25, 10)

# COMMAND ----------

# MAGIC %md
# MAGIC **Forward stepwise selection achieves the highest $R^2$ for n=13. This means that the best regression equation will contain 13 features when employing forward stepwise selection.**

# COMMAND ----------

X.columns

# COMMAND ----------

feat = sfs.k_feature_idx_
feat

# COMMAND ----------

feat_list = pd.DataFrame()
for i in feat:
    print(X.columns[i])
    feat_list = feat_list.append(pd.DataFrame([X.columns[i]]))
feat_list.reset_index(inplace=True, drop=True)
feat_list.columns = ['Features']
feat_list

# COMMAND ----------

X_reg = X[feat_list['Features']]
X_reg.head()

# COMMAND ----------

lm_best = LinearRegression().fit(X_reg, y)

# COMMAND ----------

final_coefs = pd.DataFrame(lm_best.coef_)
final_coefs = pd.concat([feat_list['Features'], final_coefs], axis = 1)
final_intercept = pd.DataFrame([lm_best.intercept_]*13)
final_coefs = pd.concat([final_coefs, final_intercept], axis = 1)
final_coefs.columns = ['Features', 'Coefficients', 'Intercept']
final_coefs

# COMMAND ----------

# MAGIC %md
# MAGIC **Therefore the best regression line for Hitters using Forward Selection will be: Salary = 92.653259 -1.565413*AtBat +  6.831929*Hits + 0.132142*HmRun -0.226508*RBI + 2.622761*Walks -0.162907*CAtBat -0.877268*CHmRun + 0.992862*CRuns + 1.009177*CRBI + 0.264327*PutOuts +  56.899972*League(=N) -109.864179*Division(=W) -11.548970*NewLeague(=N).**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Backward Stepwise Selection

# COMMAND ----------

lmb = LinearRegression()

# COMMAND ----------

X = Hitters_1.drop(columns = ['Salary', 'Names'])
y = Hitters_1.Salary

# COMMAND ----------

X.head()

# COMMAND ----------

sfs = SFS(lmf, k_features=(1,len(X.columns)), forward=False, floating=False, scoring='r2',cv=10)

# COMMAND ----------

plt.xkcd()
sfs = sfs.fit(X.values, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err', color='green')
plt.title('Forward Stepwise Selection')
plt.ylabel('R^2')
plt.grid()
fig = plt.gcf()
fig.set_size_inches(25, 10)

# COMMAND ----------

# MAGIC %md
# MAGIC **Backward stepwise selection achieves the highest $𝑅^2$ for n=8. This means that the best regression equation will contain 8 features when employing backward stepwise selection.**

# COMMAND ----------

X.columns

# COMMAND ----------

feat = sfs.k_feature_idx_[:8]
feat

# COMMAND ----------

feat_list = pd.DataFrame()
for i in feat:
    print(X.columns[i])
    feat_list = feat_list.append(pd.DataFrame([X.columns[i]]))
feat_list.reset_index(inplace=True, drop=True)
feat_list.columns = ['Features']
feat_list

# COMMAND ----------

X_reg = X[feat_list['Features']]
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

# MAGIC %md
# MAGIC **Therefore the best regression line for Hitters using Forward Selection will be: Salary = 118.235253 -1.749469*AtBat + 7.268237*Hits + 2.709401*Walks -0.108637*CAtBat +  0.810856*CRuns + 0.606283*CRBI + 0.277954*PutOuts -111.070298*Division(=W).**

# COMMAND ----------

