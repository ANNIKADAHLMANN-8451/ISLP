# Databricks notebook source
# MAGIC %md
# MAGIC In this exercise, you will further analyze the `Wage` data set considered
# MAGIC throughout this chapter.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,}'.format
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Wage.csv"
Wage = spark.read.option("header", "true").csv(url).toPandas()
Wage.set_index("SlNo", inplace=True)

str_cols = ["maritl", "race", "education", "region", "jobclass", "health", "health_ins"]
float_cols = ["logwage", "wage"]
int_cols = list(set(Wage.columns)-set(str_cols)-set(float_cols))
Wage[str_cols] = Wage[str_cols].astype(str)
Wage[float_cols] = Wage[float_cols].astype(float)
Wage[int_cols] = Wage[int_cols].astype(int)

# COMMAND ----------

Wage.head()

# COMMAND ----------

Wage.describe().round(2)

# COMMAND ----------

Wage.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Perform polynomial regression to predict `wage` using `age`. Use
# MAGIC cross-validation to select the optimal degree d for the polynomial. What degree was chosen, and how does this compare to
# MAGIC the results of hypothesis testing using ANOVA? Make a plot of
# MAGIC the resulting polynomial ft to the data.**

# COMMAND ----------

from sklearn.model_selection import KFold as KF, cross_val_score as CVS
from sklearn.linear_model import LinearRegression

# COMMAND ----------

X = Wage[['age']]
y = Wage[['wage']]

# COMMAND ----------

# selecting M, the number of components that give the lowest cross-validation error
n = len(X)
kf10 = KF(n_splits=10, shuffle=True, random_state=42)

lm = LinearRegression()
MSEdf= pd.DataFrame()
Xi = pd.DataFrame()

# calculating MSE with only the intercept through cross-validation
mse = -1*CVS(lm, np.ones((n,1)), y, cv=kf10, scoring='neg_mean_squared_error').mean()    
MSEdf = MSEdf.append([round(mse, 9)], ignore_index=True)

# calculating MSE for 15 degrees through cross-validation
for i in np.arange(1, 15):
    Xi = pd.concat([Xi, X**i], axis = 1)
    mse = -1*CVS(lm, Xi.iloc[:,:i], y, cv=kf10, scoring='neg_mean_squared_error').mean()
    MSEdf = MSEdf.append([round(mse, 9)])
    
MSEdf.reset_index(drop=True, inplace=True)
MSEdf.columns = ['MSE']
MSEdf

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(MSEdf, color='g',  linestyle='-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title('cross-validation to find optimal degree, d', fontsize=30)
plt.xlabel('order of polynomial', fontsize=20)
plt.ylabel('MSE')

# COMMAND ----------

# MAGIC %md
# MAGIC Cross-validation suggests $d$=6 to be give the smallest error and therefore is the optimal degree.

# COMMAND ----------

# MAGIC %md
# MAGIC Comparing it with $d$=4, which as suggested in R Lab 7.8.1 provides the best fit

# COMMAND ----------

df = pd.concat([y, X, X**2, X**3, X**4, X**5, X**6], axis=1)
df.columns = ['wage', 'age', 'age^2', 'age^3', 'age^4', 'age^5', 'age^6']
lmfit4 = ols("df['wage']~df['age']+df['age^2']+df['age^3']+df['age^4']", data=df).fit()
lmfit5 = ols("df['wage']~df['age']+df['age^2']+df['age^3']+df['age^4']+df['age^5']", data=df).fit()
lmfit6 = ols("df['wage']~df['age']+df['age^2']+df['age^3']+df['age^4']+df['age^5']+df['age^6']", data=df).fit()

# COMMAND ----------

anova_table = sm.stats.anova_lm(lmfit4, lmfit5, lmfit6, typ=1)
anova_table.index = anova_table.index+1
anova_table.round(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ANOVA however suggests that any degree, $d$>4 is not statistically significant (since the $p$-value>5 for $d$>4).

# COMMAND ----------

X1 = Wage['age']
X2 = X1**2
X3 = X1**3
X4 = X1**4
X5 = X1**5
X6 = X1**6
plotdf = pd.concat([y, X1, X2, X3, X4, X5, X6], axis=1)
plotdf.columns = ['wage', 'age', 'age^2', 'age^3', 'age^4', 'age^5', 'age^6']
plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot('age', 'wage', data=plotdf, fit_reg=True, ci = 95, color='y', order=6, line_kws={'color':'green'})
plt.title('wage vs poly(age, 6)', fontsize=30)
plt.xlabel('poly(age, 6)', fontsize=20)
plt.ylabel('wage', fontsize=20)

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a step function to predict `wage` using `age`, and perform crossvalidation to choose the optimal number of cuts. Make a plot of
# MAGIC the ft obtained.**

# COMMAND ----------

df = pd.concat([y, X], axis=1)
Wage_cut, bins = pd.cut(Wage.age, 4, retbins = True, right = True)
df.head()

# COMMAND ----------

# selecting M, the number of components that give the lowest cross-validation error
n = len(X)
kf10 = KF(n_splits=10, shuffle=True, random_state=42)

MSEdf= pd.DataFrame()
lm = LinearRegression()

# calculating MSE with only the intercept through cross-validation
mse = -1*CVS(lm, np.ones((n,1)), y, cv=kf10, scoring='neg_mean_squared_error').mean()    
MSEdf = MSEdf.append([round(mse, 9)], ignore_index=True)

# calculating MSE for 15 degrees through cross-validation
for i in np.arange(1, 15):
    X = pd.DataFrame(pd.cut(Wage.age, i))
    X = pd.get_dummies(X)
    y = Wage.wage
    mse = -1*CVS(lm, X, y, cv=kf10, scoring='neg_mean_squared_error').mean()
    MSEdf = MSEdf.append([round(mse, 9)])
    
MSEdf.reset_index(drop=True, inplace=True)
MSEdf.columns = ['MSE']
MSEdf

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(MSEdf, color='g',  linestyle='-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title('cross-validation to find optimal number of cuts', fontsize=30)
plt.xlabel('order of polynomial', fontsize=20)
plt.ylabel('MSE')

# COMMAND ----------

# MAGIC %md
# MAGIC From the graph, it's clear that the MSE is least at $M$=8.

# COMMAND ----------

# MAGIC %md
# MAGIC Performing regression

# COMMAND ----------

dfX = pd.DataFrame(pd.cut(Wage.age, 8))
dfX = pd.get_dummies(dfX)
dfX['age'] = Wage.age
dfy = Wage.wage
df_step = pd.concat([dfy, dfX], axis=1)
df_step.head()

# COMMAND ----------

X = df_step[df_step.columns.difference(['wage', 'age'])]
y = df_step['wage']

lmfitOLS = ols('y~X', data=df_step).fit()
lmfit = LinearRegression().fit(X, y)

# COMMAND ----------

lmfitOLS.summary()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(df_step.age, df_step.wage, color='orange', alpha=.5)
agerange = np.linspace(df_step.age.min(), df_step.age.max(), 1000)
agerange_d = pd.get_dummies(pd.cut(agerange, 8))
predictions = lmfit.predict(agerange_d)
plt.plot(agerange, predictions, color='g')
plt.title('step function', fontsize=30)
plt.xlabel('age', fontsize=20)
plt.ylabel('wage', fontsize=20)