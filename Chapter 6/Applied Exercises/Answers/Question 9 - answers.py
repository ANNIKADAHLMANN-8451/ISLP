# Databricks notebook source
# MAGIC %pip install --quiet mlxtend

# COMMAND ----------

# MAGIC %md
# MAGIC In this exercise, we will predict the number of applications received
# MAGIC using the other variables in the `College` data set.

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

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/College.csv"
College = spark.read.option("header", "true").csv(url).toPandas()
College.set_index('_c0', inplace=True)

str_cols = ["Private"]
float_cols = ["S.F.Ratio"]
int_cols = list(set(College.columns)-set(str_cols)-set(float_cols))
College[int_cols] = College[int_cols].astype(int)
College[str_cols] = College[str_cols].astype(str)
College[float_cols] = College[float_cols].astype(float)

# COMMAND ----------

College.head()

# COMMAND ----------

College.info()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.heatmap(College.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

# COMMAND ----------

# MAGIC %md
# MAGIC *There are no missing values as suggested by the heatmap above.*

# COMMAND ----------

College = pd.get_dummies(College, drop_first=True)

# COMMAND ----------

College.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Split the data set into a training set and a test set.**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X = College.drop(columns=['Apps'])
y = College['Apps']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# COMMAND ----------

X.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a linear model using least squares on the training set, and
# MAGIC report the test error obtained.**

# COMMAND ----------

lmfit = LinearRegression().fit(X_train, y_train)
lmpred = lmfit.predict(X_test)
lmcoefs = pd.Series(lmfit.coef_, index = X.columns)
lmintercept = pd.Series(lmfit.intercept_, index = ['Intercept'])
lmtable = pd.DataFrame(pd.concat([lmintercept, lmcoefs]))
lmtable.columns = ['Coefficients']
lmtable

# COMMAND ----------

lmerror = mean_squared_error(y_test, lmpred)
lmerror

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Fit a ridge regression model on the training set, with λ chosen
# MAGIC by cross-validation. Report the test error obtained.**

# COMMAND ----------

from sklearn.linear_model import Ridge, RidgeCV

# COMMAND ----------

n = 100
lambdas = (np.logspace(10, -2, num=100))

# COMMAND ----------

ridge = Ridge()
coefs = []

for k in lambdas:
    ridgelm = ridge.set_params(alpha = k).fit(X, y)
    coefs.append(ridgelm.coef_)
    
np.shape(coefs)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale('log')
plt.xlabel('lambda', fontsize = 20)
plt.ylabel('coefficients', fontsize = 20)
plt.title('coefficients as function of lambdas', fontsize = 30)
plt.legend(X)

# COMMAND ----------

# MAGIC %md
# MAGIC *This graph suggests that coefficients are likely to be larger near $\lambda$=0 (at $\lambda$=0, it will exactly be like least squares regression. So, I will need a $\lambda$-value slightly higher than that to ensure sufficient reduction in variance.*

# COMMAND ----------

# lambda = 0
ridge0 = Ridge(alpha = 0).fit(X_train, y_train)
ridgepred0 = ridge0.predict(X_test)
ridgecoefs0 = pd.Series(ridge0.coef_, index = X.columns)
ridgeintercept0 = pd.Series(ridge0.intercept_, index = ['Intercept'])
ridgetable0 = pd.DataFrame(pd.concat([ridgeintercept0, ridgecoefs0]))
ridgetable0.columns = ['Coefficients']
ridgetable0

# COMMAND ----------

ridgeerror0 = mean_squared_error(y_test, ridgepred0)
ridgeerror0

# COMMAND ----------

# MAGIC %md
# MAGIC *As explained above, setting $\lambda$=0 returns the same value of coefficients as that of least squares regression.*

# COMMAND ----------

# MAGIC %md
# MAGIC Select the best lambda using cross validation

# COMMAND ----------

ridgecv = RidgeCV(alphas = lambdas, scoring = 'neg_mean_squared_error').fit(X_train, y_train)
ridgecv.alpha_

# COMMAND ----------

# MAGIC %md
# MAGIC *Therefore, the $\lambda$ with the smallest cross-validation error is at 0.01.*

# COMMAND ----------

# lambda = 0.01
ridge001 = Ridge(alpha = ridgecv.alpha_).fit(X_train, y_train)
ridgepred001 = ridge001.predict(X_test)
ridgecoefs001 = pd.Series(ridge001.coef_, index = X.columns)
ridgeintercept001 = pd.Series(ridge001.intercept_, index = ['Intercept'])
ridgetable001 = pd.DataFrame(pd.concat([ridgeintercept001, ridgecoefs001]))
ridgetable001.columns = ['Coefficients']
ridgetable001

# COMMAND ----------

ridgeerror001 = mean_squared_error(y_test, ridgepred001)
ridgeerror001

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the MSE through ridge regression for the best value of $\lambda$ is slightly higher than that of least squares regression.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Fit a lasso model on the training set, with λ chosen by crossvalidation. Report the test error obtained, along with the number of non-zero coefcient estimates.**

# COMMAND ----------

from sklearn.linear_model import Lasso, LassoCV
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

n = 100
lambdas = (np.logspace(10, -2, num=100))

# COMMAND ----------

lasso = Lasso()
coefs = []

for k in lambdas:
    lassolm = lasso.set_params(alpha = k).fit(X, y)
    coefs.append(lassolm.coef_)
    
np.shape(coefs)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale('log')
plt.xlabel('lambda', fontsize = 20)
plt.ylabel('coefficients', fontsize = 20)
plt.title('coefficients as function of lambdas', fontsize = 30)
plt.legend(X)

# COMMAND ----------

# MAGIC %md
# MAGIC *This graph suggests that coefficients are likely to be larger near $\lambda$=0 (at $\lambda$=0, it will exactly be like least squares regression. So, I will need a $\lambda$-value slightly higher than that to ensure sufficient reduction in variance. What is interesting however, is that different coefficients reduce to 0 with increasing $\lambda$. This suggests lasso can perform variable selection in addition to variance reduction.*

# COMMAND ----------

# lambda = 0
lasso0 = Lasso(alpha = 0).fit(X_train, y_train)
lassopred0 = lasso0.predict(X_test)
lassocoefs0 = pd.Series(lasso0.coef_, index = X.columns)
lassointercept0 = pd.Series(lasso0.intercept_, index = ['Intercept'])
lassotable0 = pd.DataFrame(pd.concat([lassointercept0, lassocoefs0]))
lassotable0.columns = ['Coefficients']
lassotable0

# COMMAND ----------

lassoerror0 = mean_squared_error(y_test, lassopred0)
lassoerror0

# COMMAND ----------

# MAGIC %md
# MAGIC *As explained above, setting $\lambda$=0 returns the same value of coefficients as that of least squares regression.*

# COMMAND ----------

# MAGIC %md
# MAGIC Selecting $\lambda$ through cross-validation

# COMMAND ----------

lassocv = LassoCV(alphas = lambdas).fit(X_train, y_train)
lassocv.alpha_

# COMMAND ----------

# MAGIC %md
# MAGIC *Therefore, the $\lambda$ with the smallest cross-validation error is at 0.053366992312063016.*

# COMMAND ----------

# lambda = 0.053366992312063016
lasso005 = Lasso(alpha = lassocv.alpha_).fit(X_train, y_train)
lassopred005 = lasso005.predict(X_test)
lassocoefs005 = pd.Series(lasso005.coef_, index = X.columns)
lassointercept005 = pd.Series(lasso005.intercept_, index = ['Intercept'])
lassotable005 = pd.DataFrame(pd.concat([lassointercept005, lassocoefs005]))
lassotable005.columns = ['Coefficients']
lassotable005

# COMMAND ----------

lassoerror005 = mean_squared_error(y_test, lassopred005)
lassoerror005

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the MSE through lasso regression for the best value of $\lambda$ is slightly higher than that of least squares regression.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Fit a PCR model on the training set, with M chosen by crossvalidation. Report the test error obtained, along with the value
# MAGIC of M selected by cross-validation.**

# COMMAND ----------

from sklearn.preprocessing import scale
from sklearn.model_selection import KFold as KF
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# COMMAND ----------

pca = PCA()
X_scaled = pca.fit_transform(scale(X))
pd.DataFrame(pca.components_.T).head()

# COMMAND ----------

# selecting M, the number of components that give the lowest cross-validation error
n = len(X)
kf10 = KF(n_splits=10, shuffle=True, random_state=42)

lm = LinearRegression()
MSEdf= pd.DataFrame()

# calculating MSE with only the intercept through cross-validation
mse = -1*cross_val_score(lm, np.ones((n,1)), y.ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()    
MSEdf = MSEdf.append([round(mse, 9)])

# calculating MSE for the 17 components through cross-validation
for i in np.arange(1, 18):
    mse = -1*cross_val_score(lm, X_scaled[:,:i], y.ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()
    MSEdf = MSEdf.append([round(mse, 9)])
    
MSEdf.reset_index(drop=True, inplace=True)
MSEdf.columns = ['MSE']
MSEdf

# COMMAND ----------

# plotting MSE for each component
plt.xkcd()
plt.figure(figsize= (25, 10))
plt.plot(MSEdf)
plt.title('MSE vs number of principal components', fontsize = 30)
plt.xlabel('number of principal components', fontsize = 20)
plt.ylabel('MSE', fontsize = 20)

# COMMAND ----------

# plotting explained variance ratio
exp_var_ratio = pd.DataFrame(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))
plt.xkcd()
plt.figure(figsize= (25, 10))
plt.plot(exp_var_ratio, color='green', linestyle ='-.', marker='o', markersize=10, markerfacecolor='orange')
plt.title('explained variance ratio vs number of principal components', fontsize = 30)
plt.xlabel('number of principal components', fontsize = 20)
plt.ylabel('explained variance ratio', fontsize = 20)

# COMMAND ----------

# MAGIC %md
# MAGIC *This suggests that I will get the lowest MSE at M=17, which is the same as performing a regular least squares regression! So, I will now perform PCR on the training set and validate the model using the test set.*

# COMMAND ----------

# performing PCR on train and test data sets
pca_train = PCA()
X_scaled_train = pca_train.fit_transform(scale(X_train))
n = len(X_scaled_train)

lmtrain = LinearRegression()

kf10train = KF(n_splits=10, shuffle=True, random_state=42)
MSEdftrain= pd.DataFrame()

# calculating MSE with only the intercept through cross-validation
msetrain = -1*cross_val_score(lmtrain, np.ones((n,1)), y_train.ravel(), cv=kf10train, scoring='neg_mean_squared_error').mean()    
MSEdftrain = MSEdftrain.append([msetrain])

# calculating MSE for the 17 components through cross-validation
for i in np.arange(1, 18):
    msetrain = -1*cross_val_score(lmtrain, X_scaled_train[:,:i], y_train.ravel(), cv=kf10train, scoring='neg_mean_squared_error').mean()
    MSEdftrain = MSEdftrain.append([msetrain])
    
MSEdftrain.reset_index(drop=True, inplace=True)
MSEdftrain.columns = ['MSE']
MSEdftrain

# COMMAND ----------

# plotting MSE for each component
plt.xkcd()
plt.figure(figsize= (25, 10))
plt.plot(MSEdftrain)
plt.title('MSE vs number of principal components', fontsize = 30)
plt.xlabel('number of principal components', fontsize = 20)
plt.ylabel('MSE', fontsize = 20)

# COMMAND ----------

# plotting explained variance ratio
exp_var_ratio_scaled = pd.DataFrame(np.cumsum(np.round(pca_train.explained_variance_ratio_, decimals=4)*100))
plt.xkcd()
plt.figure(figsize= (25, 10))
plt.plot(exp_var_ratio_scaled, color='green', linestyle ='-.', marker='o', markersize=10, markerfacecolor='orange')
plt.title('explained variance ratio vs number of principal components', fontsize = 30)
plt.xlabel('number of principal components', fontsize = 20)
plt.ylabel('explained variance ratio', fontsize = 20)

# COMMAND ----------

# MAGIC %md
# MAGIC *While the MSEs themselves are somewhat reduced due to scaling, the results still suggest that I will get the best test MSE at M=17.*

# COMMAND ----------

X_scaled_test = pca.fit_transform(scale(X_test))
pcrfit17 = LinearRegression().fit(X_scaled_train, y_train)
pcrpred17 = pcrfit17.predict(X_scaled_test)

pcrerror17 = mean_squared_error(y_test, pcrpred17)
pcrerror17

# COMMAND ----------

# MAGIC %md
# MAGIC So far, PCR has produced the largest MSE amongst all models used.

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Fit a PLS model on the training set, with M chosen by crossvalidation. Report the test error obtained, along with the value
# MAGIC of M selected by cross-validation.**

# COMMAND ----------

from sklearn.cross_decomposition import PLSRegression as PLS

# COMMAND ----------

# performing PLS on train and test data sets
n = len(X_train)
kf10train = KF(n_splits=10, shuffle=True, random_state=1)

MSEdftrain= pd.DataFrame()

# calculating MSE for the 17 components through cross-validation
for k in np.arange(1, 18):
    plstrain = PLS(n_components=k)
    msetrain = -1*cross_val_score(plstrain, scale(X_train), y_train.ravel(), cv=kf10train, scoring='neg_mean_squared_error').mean()
    MSEdftrain = MSEdftrain.append([msetrain])
    
MSEdftrain.reset_index(drop=True, inplace=True)
MSEdftrain.columns = ['MSE']
MSEdftrain

# COMMAND ----------

# plotting MSE for each component
plt.xkcd()
plt.figure(figsize= (25, 10))
plt.plot(MSEdftrain)
plt.title('MSE vs number of principal components', fontsize = 30)
plt.xlabel('number of principal components', fontsize = 20)
plt.ylabel('MSE', fontsize = 20)

# COMMAND ----------

explained_variance_ratio_test = np.var(scale(X_train), axis=0) / np.sum(np.var(scale(X_train), axis=0)) # PLS does not have any method to automatically calculate explained variance ratio
EVR17 = pd.DataFrame(np.cumsum(np.round(explained_variance_ratio_test, decimals=4)*100), columns=['Explained Variance Ratio'])
plt.xkcd()
plt.figure(figsize= (25, 10))
plt.plot(EVR17, color='green', linestyle ='-.', marker='o', markersize=10, markerfacecolor='orange')
plt.title('explained variance ratio vs number of principal components', fontsize = 30)
plt.xlabel('number of principal components', fontsize = 20)
plt.ylabel('explained variance ratio', fontsize = 20)

# COMMAND ----------

# MAGIC %md
# MAGIC *The results suggest that I will get the best test MSE at M=12.*

# COMMAND ----------

plsfit12 = PLS(n_components=12).fit(scale(X_train), y_train)
plspred12 = plsfit12.predict(scale(X_test))
plserror12 = mean_squared_error(y_test, plspred12)
plserror12

# COMMAND ----------

# MAGIC %md
# MAGIC Partial least squares produces significantly lower test MSE than principal components regression.

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Comment on the results obtained. How accurately can we predict the number of college applications received? Is there much
# MAGIC diference among the test errors resulting from these five approaches?**

# COMMAND ----------

pd.options.display.float_format = '{:,}'.format
error_list = pd.DataFrame([round(lmerror, 2), round(ridgeerror001, 2), round(lassoerror005, 2), round(pcrerror17, 2), round(plserror12, 2)])
error_cols = pd.DataFrame(['linear regression', 'ridge regression', 'lasso regression', 'principal components regression', 'partial least squares'])
error_df = pd.concat([error_cols, error_list], axis=1)
error_df.columns = ['method', 'MSE']
error_df

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(error_df['MSE'], color='green', linestyle ='-.', marker='o', markersize=10, markerfacecolor='orange') # ideally I should be using a histogram, but one of the values is excessively higher than the rest which makes it difficult to view other bars
plt.xlabel('method')
plt.ylabel('MSE')
plt.title('MSE vs method')