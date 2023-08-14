# Databricks notebook source
# MAGIC %md
# MAGIC Fit some of the non-linear models investigated in this chapter to the
# MAGIC `Auto` data set. Is there evidence for non-linear relationships in this
# MAGIC data set? Create some informative plots to justify your answer.

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
import seaborn as sns; sns.set(style="ticks", color_codes=True)
%matplotlib inline

# COMMAND ----------

url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["name"]
num_cols = list(set(Auto.columns)-set(str_cols))
Auto[str_cols] = Auto[str_cols].astype(str)
Auto[num_cols] = Auto[num_cols].astype(float)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(Auto[['horsepower']]=='?', yticklabels=False, xticklabels=False, cmap='viridis')
plt.title("missing values in Auto['horsepower']", fontsize=30, color='m')

# COMMAND ----------

Auto.loc[Auto.horsepower=="?"]

# COMMAND ----------

Auto.shape

# COMMAND ----------

Auto.drop(index=[32, 126, 330, 336, 354], inplace=True)

# COMMAND ----------

Auto.shape

# COMMAND ----------

# MAGIC %md
# MAGIC *So,I have deleted the rows containing erroneous values of horsepower.*

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.heatmap(Auto[['horsepower']]=='?', yticklabels=False, xticklabels=False, cmap='viridis')
plt.title("missing values in Auto['horsepower']", fontsize=30, color='m')

# COMMAND ----------

# MAGIC %md
# MAGIC *See, no missing values!*

# COMMAND ----------

Auto.corr()

# COMMAND ----------

sns.PairGrid(Auto, hue='mpg').map(plt.scatter)

# COMMAND ----------

# MAGIC %md
# MAGIC From the pairplots, it appears that displacement, weight and acceleration have non-linear relationships with mpg.

# COMMAND ----------

# MAGIC %md
# MAGIC Polynomial

# COMMAND ----------

from sklearn.model_selection import KFold as KF, cross_val_score as CVS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC *Checking for non-linear relationship between mpg and displacement*

# COMMAND ----------

X = Auto[['displacement']]
y = Auto.mpg
df = pd.DataFrame()
MSEdf = pd.DataFrame()
SCORE = []

for k in range(0,20):
    X_k = X**k
    df = pd.concat([df, X_k], axis=1)
    df_a = np.array(df)
    lmk = LinearRegression().fit(df_a, y)
    err = pd.DataFrame([mean_squared_error(y, lmk.predict(df_a))])
    MSEdf = MSEdf.append(err)
    SCORE.append(lmk.score(df_a, y))
    
MSEdf.columns = ['MSE']
MSEdf.reset_index(drop=True, inplace=True)

SCOREdf = pd.DataFrame(SCORE)
SCOREdf.columns = ['R^2']
SCOREdf.reset_index(drop=True, inplace=True)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(MSEdf, color='g', ls = '-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title("MSE", fontsize=30, color='m')
plt.xlabel("displacement", fontsize=20, color='c')
plt.ylabel("MSE", fontsize=20, color='c')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(SCOREdf, color='g', ls = '-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title("R^2", fontsize=30, color='m')
plt.xlabel("displacement", fontsize=20, color='c')
plt.ylabel("R^2", fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC *Checking for non-linear relationship between mpg and acceleration*

# COMMAND ----------

X = Auto[['acceleration']]
y = Auto.mpg
df = pd.DataFrame()
MSEdf = pd.DataFrame()
SCORE = []

for k in range(0,25):
    X_k = X**k
    df = pd.concat([df, X_k], axis=1)
    df_a = np.array(df)
    lmk = LinearRegression().fit(df_a, y)
    err = pd.DataFrame([mean_squared_error(y, lmk.predict(df_a))])
    MSEdf = MSEdf.append(err)
    SCORE.append(lmk.score(df_a, y))
    
MSEdf.columns = ['MSE']
MSEdf.reset_index(drop=True, inplace=True)

SCOREdf = pd.DataFrame(SCORE)
SCOREdf.columns = ['R^2']
SCOREdf.reset_index(drop=True, inplace=True)

MSEdf.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(MSEdf, color='g', ls = '-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title("MSE", fontsize=30, color='m')
plt.xlabel("acceleration", fontsize=20, color='c')
plt.ylabel("MSE", fontsize=20, color='c')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(SCOREdf, color='g', ls = '-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title("R^2", fontsize=30, color='m')
plt.xlabel("acceleration", fontsize=20, color='c')
plt.ylabel("R^2", fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC *Checking for non-linear relationship between mpg and weight*

# COMMAND ----------

X = Auto[['weight']]
y = Auto.mpg
df = pd.DataFrame()
MSEdf = pd.DataFrame()
SCORE = []

for k in range(0,25):
    X_k = X**k
    df = pd.concat([df, X_k], axis=1)
    df_a = np.array(df)
    lmk = LinearRegression().fit(df_a, y)
    err = pd.DataFrame([mean_squared_error(y, lmk.predict(df_a))])
    MSEdf = MSEdf.append(err)
    SCORE.append(lmk.score(df_a, y))
    
MSEdf.columns = ['MSE']
MSEdf.reset_index(drop=True, inplace=True)

SCOREdf = pd.DataFrame(SCORE)
SCOREdf.columns = ['R^2']
SCOREdf.reset_index(drop=True, inplace=True)

MSEdf.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(MSEdf, color='g', ls = '-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title("MSE", fontsize=30, color='m')
plt.xlabel("weight", fontsize=20, color='c')
plt.ylabel("MSE", fontsize=20, color='c')

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(SCOREdf, color='g', ls = '-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title("R^2", fontsize=30, color='m')
plt.xlabel("weight", fontsize=20, color='c')
plt.ylabel("R^2", fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC In all these variables, I notice there is some form of non-linear relationship. However, the order at which they give the best fit is quite high. However, one needs to keep in mind that these scores are not cross-validated, which makes them extremely susceptible to variance. However, their non-linear relationhip with 'mpg' cannot be argued against.