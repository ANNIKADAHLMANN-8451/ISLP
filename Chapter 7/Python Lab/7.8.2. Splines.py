# Databricks notebook source
# MAGIC %pip install loess

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from scipy import stats
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,}'.format
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Wage.csv"
Wage = spark.read.option("header", "true").csv(url).toPandas()
Wage.set_index('SlNo', inplace=True)

int_cols = ["year", "age"]
float_cols = ["logwage", "wage"]
str_cols = ["maritl", "race", "education", "region", "jobclass", "health", "health_ins"]
Wage[int_cols] = Wage[int_cols].astype(int)
Wage[float_cols] = Wage[float_cols].astype(float)
Wage[str_cols] = Wage[str_cols].astype(str)

# COMMAND ----------

Wage.head()

# COMMAND ----------

Wage.describe().round(2)

# COMMAND ----------

Wage.info()

# COMMAND ----------

agegrid = np.arange(Wage['age'].min(), Wage['age'].max()).reshape(-1,1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Splines

# COMMAND ----------

from patsy import dmatrix

# COMMAND ----------

# MAGIC %md
# MAGIC **Specifying the knots in a cubic spline**

# COMMAND ----------

X1 = dmatrix("bs(AGE, knots=(25, 40, 60), degree=3, include_intercept=False)", {"AGE": Wage['age']}, return_type='dataframe')
y1 = Wage['wage']
df1 = pd.concat([y1, X1], axis=1)

# COMMAND ----------

lmfit1 = ols('y1~X1', data=df1).fit()
lmfit1.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **Degrees of freedom($df$) = 6**

# COMMAND ----------

X2 = dmatrix("bs(AGE, df=6, include_intercept=False)", {"AGE": Wage['age']}, return_type='dataframe')
y2 = Wage['wage']
df2 = pd.concat([y2, X2], axis=1)

# COMMAND ----------

lmfit2 = ols('y2~X2', data=df2).fit()
lmfit2.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **Natural spline, $df$=4**

# COMMAND ----------

X3 = dmatrix("cr(AGE, df=4)", {"AGE": Wage['age']}, return_type='dataframe')
y3 = Wage['wage']
df3 = pd.concat([y3, X3], axis=1)

# COMMAND ----------

lmfit3 = ols('y3~X3', data=df3).fit()
lmfit3.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **Comparing relative strengths of predictions of each of the aforementioned methods**

# COMMAND ----------

from sklearn.linear_model import LinearRegression
lmfit1_skl = LinearRegression().fit(X1, y1)
lmfit2_skl = LinearRegression().fit(X2, y2)
lmfit3_skl = LinearRegression().fit(X3, y3)

# COMMAND ----------

Xpred1 = dmatrix("bs(agegrid, knots=(25,40,60), include_intercept=False)", {"agegrid": agegrid}, return_type='dataframe')
Xpred2 = dmatrix("bs(agegrid, df=6, include_intercept=False)", {"age_grid": agegrid}, return_type='dataframe')
Xpred3 = dmatrix("cr(agegrid, df=4)", {"agegrid": agegrid}, return_type='dataframe')

# COMMAND ----------

lmpred1 = lmfit1_skl.predict(Xpred1)
lmpred2 = lmfit2_skl.predict(Xpred2)
lmpred3 = lmfit3_skl.predict(Xpred3)

# COMMAND ----------

# plotting all predictions
plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(Wage['age'], Wage['wage'], facecolor='y', alpha=0.5)
plt.plot(agegrid, lmpred1, color='r', label='Specifying the knots in a cubic spline')
plt.plot(agegrid, lmpred2, color='g', label='Degrees of freedom(df)=6')
plt.plot(agegrid, lmpred3, color='b', label='Natural spline, df=4')
[plt.vlines(i , 0, 350, linestyles='dashed', lw=2, colors='k') for i in [25,40,60]]
plt.legend()
plt.xlabel('age', fontsize=20, color='c')
plt.ylabel('spline predictions', fontsize=20, color='c')
plt.title('spline predictions vs age', fontsize=30, color='m')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Local regression

# COMMAND ----------

import loess
from statsmodels.nonparametric.smoothers_lowess import lowess

# COMMAND ----------

lX = Wage.age
ly = Wage.wage

# COMMAND ----------

lsfit02 = lowess(ly, lX, frac=0.2, return_sorted=True)
lsfit02

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(Wage.age, Wage.wage, facecolor='lightgrey')
plt.plot(lsfit02[:3000,0], lsfit02[:3000,1], color='g')
plt.title('wage as a function of age using local regression (span=0.2)', fontsize=30, color='m')
plt.xlabel('age', fontsize=20, color='c')
plt.ylabel('wage', fontsize=20, color='c')

# COMMAND ----------

lsfit05 = lowess(ly, lX, frac=0.5)
lsfit05

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(Wage.age, Wage.wage, facecolor='lightgrey')
plt.plot(lsfit05[:3000,0], lsfit05[:3000,1], color='g')
plt.title('wage as a function of age using local regression (span=0.5)', fontsize=30, color='m')
plt.xlabel('age', fontsize=20, color='c')
plt.ylabel('wage', fontsize=20, color='c')

# COMMAND ----------

lsfit09 = lowess(ly, lX, frac=0.9)
lsfit09

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(Wage.age, Wage.wage, facecolor='lightgrey')
plt.plot(lsfit09[:3000,0], lsfit09[:3000,1], color='g')
plt.title('wage as a function of age using local regression (span=0.9)', fontsize=30, color='m')
plt.xlabel('age', fontsize=20, color='c')
plt.ylabel('wage', fontsize=20, color='c')

# COMMAND ----------

lsfit01 = lowess(ly, lX, frac=0.1)
lsfit01

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(Wage.age, Wage.wage, facecolor='lightgrey')
plt.plot(lsfit01[:3000,0], lsfit01[:3000,1], color='g')
plt.title('wage as a function of age using local regression (span=0.1)', fontsize=30, color='m')
plt.xlabel('age', fontsize=20, color='c')
plt.ylabel('wage', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC **As we can see, larger the span, smoother the fit.**