# Databricks notebook source
# MAGIC %pip install --quiet mlxtend

# COMMAND ----------

# MAGIC %md
# MAGIC We will now try to predict per capita crime rate in the `Boston` data
# MAGIC set. Note, this notebook was modified from the original repo as the `normalize` parameter in some of the model instance functions has since been removed.

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

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index("SlNo", inplace=True)

# COMMAND ----------

Boston.head()

# COMMAND ----------

Boston = pd.get_dummies(Boston, columns =['chas'], drop_first=True)

# COMMAND ----------

X = Boston.drop(columns='crim')
y = Boston['crim']

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Try out some of the regression methods explored in this chapter,
# MAGIC such as best subset selection, the lasso, ridge regression, and
# MAGIC PCR. Present and discuss results for the approaches that you
# MAGIC consider.**

# COMMAND ----------

# MAGIC %md
# MAGIC Lasso regression

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

# splitting the Boston dataset into training and test data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# COMMAND ----------

# lambda = 0
lasso0 = Lasso(alpha = 0).fit(X_train, y_train)
lassopred0 = lasso0.predict(scale(X_test))
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
# MAGIC *Therefore, the $\lambda$ with the smallest cross-validation error is at 0.013219411484660288.*

# COMMAND ----------

# lambda = 0.013219411484660288
lasso0013 = Lasso(alpha = lassocv.alpha_).fit(X_train, y_train)
lassopred0013 = lasso0013.predict(scale(X_test))
lassocoefs0013 = pd.Series(lasso0013.coef_, index = X.columns)
lassointercept0013 = pd.Series(lasso0013.intercept_, index = ['Intercept'])
lassotable0013 = pd.DataFrame(pd.concat([lassointercept0013, lassocoefs0013]))
lassotable0013.columns = ['Coefficients']
lassotable0013

# COMMAND ----------

lassoerror0013 = mean_squared_error(y_test, lassopred0013)
lassoerror0013

# COMMAND ----------

# MAGIC %md
# MAGIC Ridge regression

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
ridgepred0 = ridge0.predict(scale(X_test))
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
# MAGIC *As explained above, setting $\lambda$=0 returns the same value of coefficients as that of least squares regression. What's interesting to note that the coefficients and the MSE through ridge regression is the same as those through lasso
# MAGIC regression at $\lambda$=0.*

# COMMAND ----------

# MAGIC %md
# MAGIC Selecting $\lambda$ through cross-validation

# COMMAND ----------

ridgecv = RidgeCV(alphas = lambdas, scoring = 'neg_mean_squared_error').fit(X_train, y_train)
ridgecv.alpha_

# COMMAND ----------

# MAGIC %md
# MAGIC *Therefore, the $\lambda$ with the smallest cross-validation error is at 0.07054802310718632.*

# COMMAND ----------

# lambda = 0.07054802310718632
ridge007 = Ridge(alpha = ridgecv.alpha_).fit(X_train, y_train)
ridgepred007 = ridge007.predict(scale(X_test))
ridgecoefs007 = pd.Series(ridge007.coef_, index = X.columns)
ridgeintercept007 = pd.Series(ridge007.intercept_, index = ['Intercept'])
ridgetable007 = pd.DataFrame(pd.concat([ridgeintercept007, ridgecoefs007]))
ridgetable007.columns = ['Coefficients']
ridgetable007

# COMMAND ----------

ridgeerror007 = mean_squared_error(y_test, ridgepred007)
ridgeerror007

# COMMAND ----------

# MAGIC %md
# MAGIC Principal components regression

# COMMAND ----------

from sklearn.preprocessing import scale
from sklearn.model_selection import KFold as KF, cross_val_score
from sklearn.decomposition import PCA

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

# calculating MSE for the 20 components through cross-validation
for i in np.arange(1, 21):
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
# MAGIC *This suggests that I will get the lowest MSE at M=12 and flattens thereafter.*

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

# calculating MSE for the 20 components through cross-validation
for i in np.arange(1, 21):
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
# MAGIC *The two graphs above are quite interesting, The first graph suggests that the least MSE occurs at M=3. But, if we check the explained variance ratio, only  76.82% of the variance is explained by M=3 as comparied to 100% for M=12. So, it seems right that the ideal number of principal components is M=12.*

# COMMAND ----------

y_test

# COMMAND ----------

pca_test = PCA(n_components=12)
X_scaled_test = pca_test.fit_transform(scale(X_test))
pcrfit12 = LinearRegression().fit(X_scaled_train, y_train)
y_test_pd = pd.DataFrame({'y': y_test})
X_scaled_test = np.concatenate((X_scaled_test, y_test_pd), axis=1)
pcrpred12 = pcrfit12.predict(X_scaled_test)

pcrerror12 = mean_squared_error(y_test, pcrpred12)
pcrerror12

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Propose a model (or set of models) that seem to perform well on
# MAGIC this data set, and justify your answer. Make sure that you are
# MAGIC evaluating model performance using validation set error, crossvalidation, or some other reasonable alternative, as opposed to
# MAGIC using training error.**

# COMMAND ----------

errordf = pd.DataFrame([lassoerror0013, ridgeerror007, pcrerror12], ['lasso', 'ridge', 'pcr'])
errordf.columns = ['error']
errordf

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, I will choose the lasso because it generates the least mean squared error on the test set.

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Does your chosen model involve all of the features in the data
# MAGIC set? Why or why not?**

# COMMAND ----------

lassotable0013

# COMMAND ----------

# MAGIC %md
# MAGIC No, it does not incorporate all features. Therefore, the equation with lasso will be $Y$ = $1.692065$ + $0.001496AGE$ - $0.203664DIS$ + $0.478376RAD$ + $0.000344TAX$ - $0.007527B$ + $0.041371LSTAT$ - $0.556786CHAS(=1)$.

# COMMAND ----------

lassotable0

# COMMAND ----------

# MAGIC %md
# MAGIC This is because rest of the features are approximately equal to 0. Therefore, lasso punishes these features by reducing them to exactly 0.

# COMMAND ----------

