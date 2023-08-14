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
from numpy import linalg

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
Hitters_1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lasso Regression

# COMMAND ----------

from sklearn.linear_model import Lasso

# COMMAND ----------

X = Hitters_1.drop(columns = ['Salary', 'Names'])
y = Hitters_1.Salary

# COMMAND ----------

# standardisation
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# COMMAND ----------

n = 100
lambdas = (np.logspace(10, -2, num=100))

# COMMAND ----------

coefs = []
MSE = []
for k in lambdas:
    lassomod = Lasso(alpha=k, fit_intercept=True).fit(X_scaled,y)
    coefs.append(lassomod.coef_)
    MSE.append(mean_squared_error(y, lassomod.predict(X_scaled)))

# COMMAND ----------

coefPD = pd.DataFrame(coefs)

# COMMAND ----------

coefPD.head()

# COMMAND ----------

coefPD.shape

# COMMAND ----------

# MAGIC %md
# MAGIC **The book finds the shape of the coefficient matrix to be (100, 20). This actually makes sense because the dataframe above
# MAGIC does not contain intercept. So, I will add the intercept at the beginning of each row.**

# COMMAND ----------

lassomod.intercept_

# COMMAND ----------

coefPD = pd.concat([pd.DataFrame([lassomod.intercept_]*100), coefPD], axis=1)
coefPD.columns = ['Intercept', 'AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Walks', 'Years', 'CAtBat',
       'CHits', 'CHmRun', 'CRuns', 'CRBI', 'CWalks', 'PutOuts', 'Assists',
       'Errors', 'League_N', 'Division_W', 'NewLeague_N']
coefPD.shape

# COMMAND ----------

lambdas[49] # Python starts counting at 0. This will be equal to ridge.mod$lambda[50] in the book since R starts counting rows at 1.

# COMMAND ----------

coefPD.iloc[49]

# COMMAND ----------

lambdas[59]

# COMMAND ----------

coefPD.iloc[59]

# COMMAND ----------

plt.xkcd()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,10))

# indexing the five largest coefficients
idx = np.argpartition(np.abs(coefs[-1]), -5)[-5:]

# standardized coefficients vs lambdas
ax1.plot(lambdas, coefs)
ax1.set_xscale('log')
ax1.set_xlabel('lambda')
ax1.set_ylabel('standardized soefficients')
ax1.set_title('standardised coefficients vs lambdas')
ax1.legend(np.array(ax1.get_lines())[idx], X.columns[idx])

# standardized coefficients vs l2 norms
l2norm = linalg.norm(coefs[-1])
l2coefs = linalg.norm(coefs/l2norm, axis=1)
ax2.plot(l2coefs, coefs)
ax2.set_xlabel('l2 norm of ridge coefficients / l2 norm of least squares coefficients')
ax2.set_ylabel('standardized coefficients')
ax2.set_title('standardised coefficients vs l2 norms')
ax2.legend(np.array(ax2.get_lines())[idx], X.columns[idx]);

# 'Mean Square Error(MSE) vs lambdas
ax3.plot(lambdas, MSE)
ax3.set_xscale('log')
ax3.set_xlabel('lambda')
ax3.set_ylabel('Mean Square Error(MSE)')
ax3.set_title('Mean Square Error(MSE) vs lambdas');

# COMMAND ----------

# MAGIC %md
# MAGIC **It is quite interesting to note how different variables reach 0 at different intervals. This is better visualised in the
# MAGIC middle plot where different predictors enter the model one by one at different intervals.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split dataset into training and test dataset (and standardise them)

# COMMAND ----------

from sklearn.model_selection import train_test_split
X = Hitters_1.drop(columns = ['Salary', 'Names'])
y = Hitters_1.Salary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# COMMAND ----------

# standardisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# MSE with only the intercept
lmMSEintercept = np.mean(pow((y_train.mean()-y_test), 2))
print("MSE with only the intercept: ", lmMSEintercept)

# MSE for lambda = 0; this is similar to least squares linear regression
lmlasso0 = Lasso(alpha=0, fit_intercept=True).fit(X_train_scaled, y_train)
lmpredict0 = lmlasso0.predict(X_test_scaled)
lmMSE0 = mean_squared_error(y_test, lmpredict0)
print("MSE at lambda = 0: ", lmMSE0)

# MSE for lambda = 1
lmridge1 = Lasso(alpha=1, fit_intercept=True).fit(X_train_scaled, y_train)
lmpredict1 = lmridge1.predict(X_test_scaled)
lmMSE1 = mean_squared_error(y_test, lmpredict1)
print("MSE at lambda = 1: ", lmMSE1)

# MSE for lambda = 4
lmlasso4 = Lasso(alpha=4, fit_intercept=True).fit(X_train_scaled, y_train)
lmpredict4 = lmlasso4.predict(X_test_scaled)
lmMSE4 = mean_squared_error(y_test, lmpredict4)
print("MSE at lambda = 4: ", lmMSE4)

# MSE for lambda = pow(10, 10)
lmlasso1010 = Lasso(alpha=pow(10, 10), fit_intercept=True).fit(X_train_scaled, y_train)
lmpredict1010 = lmlasso1010.predict(X_test_scaled)
lmMSE1010 = mean_squared_error(y_test, lmpredict1010)
print("MSE at lambda = 10^10: ", lmMSE1010)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lasso regression with cross-validation

# COMMAND ----------

# finding the best lambda using CV
from sklearn.linear_model import LassoCV
lmlassoCV = LassoCV(alphas=lambdas, cv=10).fit(X_train_scaled, y_train)
lmlassoCValpha = lmlassoCV.alpha_
print("Best lambda: ", lmlassoCValpha)

# COMMAND ----------

# performing lasso regression using best lambda
lmlasso = Lasso(alpha=lmlassoCValpha, fit_intercept=True).fit(X_train_scaled, y_train)
lmlasso_MSE = mean_squared_error(y_test, lmlasso.predict(X_test_scaled))
print('MSE for best lambda: ', lmlasso_MSE)

# COMMAND ----------

intercept_list = pd.DataFrame([lmlasso.intercept_]*19)
coef_list = pd.concat([intercept_list, pd.DataFrame([lmlasso.coef_]).T], axis = 1)
coef_list.reset_index(inplace=True, drop=True)
coef_list.columns = ['Intercept', 'Coefficients']
coef_list

# COMMAND ----------

# MAGIC %md
# MAGIC **Two points of note:**<br>
# MAGIC **1. The test errors of lasso regression are comparable to those of ridge regression.**
# MAGIC <br>
# MAGIC **2. 9 out of 18 coefficients are exactly 0. This suggests that lasso regression can perform feature selection as well as variance reduction.**