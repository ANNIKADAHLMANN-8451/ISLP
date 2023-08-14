# Databricks notebook source
# MAGIC %md
# MAGIC This question uses the variables `dis` (the weighted mean of distances
# MAGIC to fve Boston employment centers) and `nox` (nitrogen oxides concentration in parts per 10 million) from the `Boston` data. We will treat
# MAGIC `dis` as the predictor and `nox` as the response.

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
from sklearn.metrics import r2_score, mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_linear_regression as PLS
from sklearn.preprocessing import scale

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index("SlNo", inplace=True)

# COMMAND ----------

Boston = pd.get_dummies(Boston, columns =['chas'], drop_first=True)

# COMMAND ----------

Boston.head()

# COMMAND ----------

Boston.shape

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Use the `LinearRegression()` function from `sklearn` module to ft a
# MAGIC cubic polynomial regression to predict `nox` using `dis`. Report the
# MAGIC regression output, and plot the resulting data and polynomial
# MAGIC fts.**

# COMMAND ----------

X = Boston['dis']
y = Boston['nox']

# COMMAND ----------

from sklearn.model_selection import KFold as KF, cross_val_score as CVS
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm

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
MSEdf.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(MSEdf, color='g',  linestyle='-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title('cross-validation to find optimal degree, d', fontsize=30)
plt.xlabel('order of polynomial', fontsize=20)
plt.ylabel('MSE')

# COMMAND ----------

# MAGIC %md
# MAGIC Cross-validation suggests $d$=3 to be give the smallest error and therefore is the optimal degree.

# COMMAND ----------

# MAGIC %md
# MAGIC Comparing it with $d$=4 and $d$=2

# COMMAND ----------

df = pd.concat([y, X, X**2, X**3, X**4], axis=1)
df.columns = ['nox', 'dis', 'dis^2', 'dis^3', 'dis^4']
lmfit2 = ols("df['nox']~df['dis']+df['dis^2']", data=df).fit()
lmfit3 = ols("df['nox']~df['dis']+df['dis^2']+df['dis^3']", data=df).fit()
lmfit4 = ols("df['nox']~df['dis']+df['dis^2']+df['dis^3']+df['dis^4']", data=df).fit()

# COMMAND ----------

anova_table = sm.stats.anova_lm(lmfit2, lmfit3, lmfit4, typ=1)
anova_table.index = anova_table.index+2
anova_table.round(2)

# COMMAND ----------

# MAGIC %md
# MAGIC I notice that at $d$=4, the $p$-value>5%. Therefore, it's not significant to have $d$>3.

# COMMAND ----------

X1 = Boston['dis']
X2 = X1**2
X3 = X1**3
plotdf = pd.concat([y, X1, X2, X3], axis=1)
plotdf.columns = ['nox', 'dis', 'dis^2', 'dis^3']
plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot('dis', 'nox', data=plotdf, fit_reg=True, ci = 95, color='y', order=3, line_kws={'color':'green'})
plt.title('NOX vs poly(DIS, 3)', fontsize=30)
plt.xlabel('poly(DIS, 3)', fontsize=20)
plt.ylabel('NOX', fontsize=20)

# COMMAND ----------

lmfit3.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Plot the polynomial fts for a range of diferent polynomial
# MAGIC degrees (say, from 1 to 10), and report the associated residual
# MAGIC sum of squares.**

# COMMAND ----------

X = Boston[['dis']]
y = Boston['nox']

n = len(X)

df = pd.DataFrame()
RSSdf = pd.DataFrame()

for k in range(0,21):
    X_k = X**k
    df = pd.concat([df, X_k], axis=1)
    df_a = np.array(df)
    lmk = LinearRegression().fit(df_a, y)
    err = pd.DataFrame([mean_squared_error(y, lmk.predict(df_a))*n]) # multiplying MSE by n gives RSS
    RSSdf = pd.concat([RSSdf, err], axis=0)

RSSdf.columns = ['RSS']
RSSdf.reset_index(drop=True, inplace=True)
RSSdf.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(RSSdf, color='g', ls = '-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title("RSS as a function of degrees, d", fontsize=30, color='m')
plt.xlabel("degrees, d", fontsize=20, color='c')
plt.ylabel("RSS", fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC Without cross-validation, I get the lowest RSS at $d$=16. However, using such high-degree polynomial regression risks the model overfitting test results (I have achieved this RSS by predicting on the training data set).

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Use the `LinearRegression()` function from the `sklearn` module to fit a regression spline to predict `nox` using `dis`. Report the output for
# MAGIC the fit using four degrees of freedom. How did you choose the
# MAGIC knots? Plot the resulting fit.**

# COMMAND ----------

from patsy import dmatrix

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
df = pd.concat([y, X], axis=1)
sns.distplot(X, color='g', bins=30, kde=True, rug=True, hist_kws={"color":"yellow"}, kde_kws={"color":"green"})
plt.title("distribution of 'dis'", fontsize=30, color='m')
plt.xlabel('dis', fontsize=20, color='c')

# COMMAND ----------

# MAGIC %md
# MAGIC *I notice that NOX has the most significant change in distribution at 2 and is fairly smooth otherwise. So, I choose the knots at (2, 4, 6, 8)*

# COMMAND ----------

X1 = dmatrix("bs(DIS, knots=(2, 4, 6, 8), degree=3, include_intercept=False)", {"DIS": X}, return_type='dataframe')
y1 = y
df1 = pd.concat([y1, X1], axis=1)

# COMMAND ----------

lmfit1 = ols('y1~X1', data=df1).fit()
lmfit1.summary()

# COMMAND ----------

lmfit1_skl = LinearRegression().fit(X1, y1)
DISgrid = np.arange(Boston.dis.min(), Boston.dis.max()).reshape(-1,1)
Xpred1 = dmatrix("bs(DIS, knots=(2, 4, 6, 8), degree=3, include_intercept=False)", {"DIS": DISgrid}, return_type='dataframe')
lmpred1 = lmfit1_skl.predict(Xpred1)

# COMMAND ----------

# plotting all predictions
plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(X, y, facecolor='y', alpha=0.5)
plt.plot(DISgrid, lmpred1, color='r', label='Specifying the knots in a spline')
[plt.vlines(i , 0, 12, linestyles='dashed', lw=2, colors='k') for i in [2, 4, 6, 8]]
plt.legend()
plt.ylim(ymin=0,ymax=1)
plt.xlabel('DIS', fontsize=20, color='c')
plt.ylabel('spline predictions for NOX', fontsize=20, color='c')
plt.title('spline predictions for NOX vs DIS', fontsize=30, color='m')

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now fit a regression spline for a range of degrees of freedom, and
# MAGIC plot the resulting fits and report the resulting RSS. Describe the
# MAGIC results obtained.**

# COMMAND ----------

X = Boston[['dis']]
y = Boston['nox']
MSEdf = pd.DataFrame()
r2df = pd.DataFrame()

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(X, y, facecolor='y', alpha=0.5)
[plt.vlines(i , 0, 12, linestyles='dashed', lw=2, colors='k') for i in [2, 4, 6, 8, 10, 12]]
plt.ylim(ymin=0,ymax=1)
plt.xlabel('DIS', fontsize=20, color='c')
plt.ylabel('spline predictions for NOX', fontsize=20, color='c')
plt.title('spline predictions for NOX vs DIS', fontsize=30, color='m')
for k in range(3,20):
    Xk = dmatrix("bs(DIS, df=k)", {"DIS": X}, return_type='dataframe')
    yk = y
    dfX = pd.concat([yk, Xk], axis=1)
    lmfit_skl = LinearRegression().fit(Xk, yk)
    DISgrid = np.arange(Boston.dis.min(), Boston.dis.max()).reshape(-1,1)
    Xpredk = dmatrix(f"bs(DIS, df=k)", {"DIS": DISgrid}, return_type='dataframe')
    lmpredk = lmfit_skl.predict(Xpredk)
    lmpredk1 = lmfit_skl.predict(Xk)
    err = pd.DataFrame([mean_squared_error(y, lmpredk1)])
    MSEdf = MSEdf.append(err)
    r2 = pd.DataFrame([r2_score(y, lmpredk1)])
    r2df = r2df.append(r2)
    plt.plot(DISgrid, lmpredk, label=k)
    plt.legend()
    
MSEdf.columns = ['MSE']
MSEdf.reset_index(drop=True, inplace=True)
MSEdf.index = MSEdf.index+3

r2df.columns = ['MSE']
r2df.reset_index(drop=True, inplace=True)
MSEdf.index = MSEdf.index+3

# COMMAND ----------

# MAGIC %md
# MAGIC I have plotted regression splines from degrees, $d$=3 to $d$=20. It's easy to see that the spline overfits the data in higher order.

# COMMAND ----------

plt.xkcd()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,10))

ax1.plot(MSEdf,a ls='-.', color='g', marker='o', markersize=10, markerfacecolor='orange')
ax1.set_xlabel('degrees', fontsize=20, color='c')
ax1.set_ylabel('MSE', fontsize=20, color='c')
ax1.set_title('MSE for degrees of DIS', color='m', fontsize=30)

ax2.plot(r2df, ls='-.', color='g', marker='o', markersize=10, markerfacecolor='orange')
ax2.set_xlabel('degrees', fontsize=20, color='c')
ax2.set_ylabel('R^2', fontsize=20, color='c')
ax2.set_title('R^2 for degrees of DIS', color='m', fontsize=30)

# COMMAND ----------

# MAGIC %md
# MAGIC The MSE decreases monotonically and then the rate of decrease slows down near $d$=20. Likewise, the $R^2$ increases monotonically and then the rate of increase slows down near $d$=20

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Perform cross-validation or another approach in order to select
# MAGIC the best degrees of freedom for a regression spline on this data.
# MAGIC Describe your results.**

# COMMAND ----------

from sklearn.pipeline import make_pipeline
X = Boston.dis.values.reshape(-1, 1)
y = Boston.dis.values

# COMMAND ----------

# selecting M, the number of components that give the lowest cross-validation error
n = len(X)
kf10 = KF(n_splits=10, shuffle=True, random_state=42)

lm = LinearRegression()
MSEdf= pd.DataFrame()
Xk = pd.DataFrame()

# calculating MSE for 15 degrees through cross-validation
for k in np.arange(3, 35):
    Xk = dmatrix("bs(DIS, df=k)", {"DIS": X}, return_type='dataframe')
    mse = -1*CVS(lm, Xk.iloc[:,:k], y, cv=kf10, scoring='neg_mean_squared_error').mean()
    MSEdf = MSEdf.append([round(mse, 9)])
    
MSEdf.reset_index(drop=True, inplace=True)
MSEdf.columns = ['MSE']
MSEdf.index = MSEdf.index + 3
MSEdf.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(MSEdf, color='g',  linestyle='-.', marker='o', markerfacecolor='orange', markersize=10)
plt.title('cross-validation to find optimal degree, d', fontsize=30)
plt.xlabel('order of polynomial', fontsize=20)
plt.ylabel('MSE')

# COMMAND ----------

# MAGIC %md
# MAGIC The lowest MSE occurs at $d$=14 and then increases over increasing degrees. Therefore, $d$=10 is the optimal degrees of freedom.

# COMMAND ----------

