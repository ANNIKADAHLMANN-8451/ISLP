# Databricks notebook source
# MAGIC %md
# MAGIC ### Preprocessing - I

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

from sklearn.model_selection import train_test_split

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Portfolio.csv"
Portfolio = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Portfolio.set_index('_c0', inplace=True)
Portfolio.index.names = ['Index']

# COMMAND ----------

Portfolio.head()

# COMMAND ----------

covmatrix = Portfolio.cov()

# COMMAND ----------

covmatrix

# COMMAND ----------

covmatrix.iloc[0][1]

# COMMAND ----------

# MAGIC %md
# MAGIC *Okay cool!*

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estimating the Accuracy of a Statistic of Interest through Bootstrap

# COMMAND ----------

def alphafn(data, index):
    X = data.X[index]
    Y = data.Y[index]
    return round((Y.var() - covmatrix.iloc[0][1]) / (X.var() + Y.var() - 2*covmatrix.iloc[0][1]), 3)

# COMMAND ----------

idx = list(range(1,101))

# COMMAND ----------

true_alpha = alphafn(Portfolio, idx)
true_alpha

# COMMAND ----------

from sklearn.utils import resample

# COMMAND ----------

estimate_alpha = pd.DataFrame()
for i in range(1,1001): # this is equivalent to constructing a new bootstrap data set and recomputing alpha_hat based on the new data set and generating 1,000 bootstrap estimates 
    bootstrap_estimate = alphafn(Portfolio, resample(idx, replace=True))
    estimate_alpha = estimate_alpha.append([bootstrap_estimate])    

# COMMAND ----------

estimate_alpha.reset_index(drop=True, inplace=True)
estimate_alpha.columns = ['Alpha Estimate']
estimate_alpha.head()

# COMMAND ----------

estimate_alpha.shape

# COMMAND ----------

# MAGIC %md
# MAGIC *We see here that we have generated 1,000 estimates of alpha.*

# COMMAND ----------

std_err = np.sqrt(np.sum(pow(estimate_alpha - (np.sum(estimate_alpha) / 1000), 2))/999) # check formula for standard error in page 189 of ISLR

# COMMAND ----------

std_err

# COMMAND ----------

# MAGIC %md
# MAGIC **Therefore, the estimate of $\alpha$-hat using the original data is 0.576, and that the bootstrap estimate for SE($\alpha$-hat)
# MAGIC is 0.091405.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing - II

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

from sklearn.model_selection import train_test_split

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["name"]
num_cols = list(set(Auto.columns) - set(str_cols))
Auto[str_cols] = Auto[str_cols].astype(str)
Auto[num_cols] = Auto[num_cols].astype(float)

# COMMAND ----------

Auto.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estimating the Accuracy of a Linear Regression Model through Bootstrap

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

def bootfn(data, index):
    X = data[['horsepower']]
    y = data['mpg']
    X_train = X.iloc[index]
    y_train = y.iloc[index]
    lmfit = LinearRegression().fit(X_train, y_train)
    return round(lmfit.intercept_, 3), np.round(lmfit.coef_, 3)

# COMMAND ----------

idx = list(range(1,393))

# COMMAND ----------

bootfn(Auto, idx)

# COMMAND ----------

# MAGIC %md
# MAGIC **Creating bootstrap estimates**

# COMMAND ----------

bootfn(Auto, resample(idx, replace=True))

# COMMAND ----------

bootfn(Auto, resample(idx, replace=True))

# COMMAND ----------

# MAGIC %md
# MAGIC *As we can see, bootstrapping generates different estimates for both the intercept and slope each time through random
# MAGIC sampling.*

# COMMAND ----------

estimate_coefficients = pd.DataFrame()
for i in range(1,1001): # this is equivalent to constructing a new bootstrap data set and recomputing alpha_hat based on the new data set and generating 1,000 bootstrap estimates 
    coef_estimate = bootfn(Auto, resample(idx, replace=True))
    estimate_coefficients = estimate_coefficients.append([coef_estimate])   

# COMMAND ----------

estimate_coefficients.reset_index(drop=True, inplace=True)
estimate_coefficients.columns = ['Intercept', 'Slope Term']
estimate_coefficients.head()

# COMMAND ----------

estimate_coefficients.shape

# COMMAND ----------

# MAGIC %md
# MAGIC *We see here that we have generated 1,000 estimates of intercepts and slope terms.*

# COMMAND ----------

std_err_st = np.sqrt(np.sum(pow(estimate_coefficients['Slope Term'] - (np.sum(estimate_coefficients['Slope Term']) / 1000), 2))/999)

# COMMAND ----------

std_err_intercept = np.sqrt(np.sum(pow(estimate_coefficients['Intercept'] - (np.sum(estimate_coefficients['Intercept']) / 1000), 2))/999)

# COMMAND ----------

print("Bootsrapped Intercept Standard Error: ", round(std_err_intercept, 4), "Bootstrapped Slope Term Standard Error: ", np.round(std_err_st, 4))

# COMMAND ----------

# MAGIC %md
# MAGIC **Conducting simple linear regression**

# COMMAND ----------

import statsmodels.api as sm

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** scikit-learn does not have modules for inference. Hence, I am importing statsmodels to generate inferential statistics to get standard errors from simple linear regression

# COMMAND ----------

import statsmodels.api as sm

# COMMAND ----------

X1 = Auto[['horsepower']]
X1 = sm.add_constant(X1)
y1 = Auto['mpg']
lmfit1 = LinearRegression().fit(X1, y1)

# COMMAND ----------

ols = sm.OLS(y1, X1).fit()

# COMMAND ----------

ols.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interestingly, note here that the standard errors generated through the bootstrap model seem way off than those from 
# MAGIC the simple linear regression. This is because the bootstrap model does not presume assumptions for standard errors like
# MAGIC they do in simple linear regression. Therfore, bootstrap estimates of standard errors are more likely to be accurate than
# MAGIC those of simple linear regression. For more information, I recommend reading page 196 of ISLR.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estimating the Accuracy of a Quadratic Regression Model through Bootstrap

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

def bootfn(df, index):
    for i in range(1, 3):
        MSE = 0
        X = df[['horsepower']]
        X_ = pd.DataFrame(PF(i).fit_transform(X))
        X_.drop(columns=0, inplace=True)
        y = df[['mpg']]
        X_train = X_.iloc[index]
        y_train = y.iloc[index]
        lmfit = LinearRegression().fit(X_train, y_train)
    return lmfit.intercept_, lmfit.coef_

# COMMAND ----------

bootfn(Auto, idx)

# COMMAND ----------

estim = bootfn(Auto, resample(idx, replace=True))

# COMMAND ----------

estim

# COMMAND ----------

df1 = pd.DataFrame()
df1 = pd.concat([pd.DataFrame([estim[1].reshape(2,-1)[0]]), pd.DataFrame([estim[1].reshape(2,-1)[1]])], axis=1)

# COMMAND ----------

estimate_coef = pd.DataFrame()

# COMMAND ----------

for i in range(1,1001):
    coefs = bootfn(Auto, resample(idx, replace=True))
    estimate_coef = estimate_coef.append(pd.concat([pd.DataFrame(coefs[0]), pd.DataFrame([coefs[1].reshape(2,-1)[0]]), pd.DataFrame([coefs[1].reshape(2,-1)[1]])], axis=1))

# COMMAND ----------

estimate_coef.reset_index(drop=True, inplace=True)
estimate_coef.columns = ['Intercept', 'hp', 'hp^2']

# COMMAND ----------

estimate_coef.head()

# COMMAND ----------

estimate_coef.shape

# COMMAND ----------

std_err_intercept = np.sqrt(np.sum(pow(estimate_coef['Intercept'] - (np.sum(estimate_coef['Intercept']) / 1000), 2))/999)

# COMMAND ----------

std_err_hp = np.sqrt(np.sum(pow(estimate_coef['hp'] - (np.sum(estimate_coef['hp']) / 1000), 2))/999)

# COMMAND ----------

std_err_hp2 = np.sqrt(np.sum(pow(estimate_coef['hp^2'] - (np.sum(estimate_coef['hp^2']) / 1000), 2))/999)

# COMMAND ----------

print("SE Intercept: ", round(std_err_intercept, 4), "SE HP: ", round(std_err_hp, 4), "SE HP^2: ", round(std_err_hp2, 4))

# COMMAND ----------

# MAGIC %md
# MAGIC **Linear Regression**

# COMMAND ----------

import statsmodels.api as sm

# COMMAND ----------

X1 = Auto[['horsepower']]
X2 = pow(X1, 2)
X3 = pd.concat([X1, X2], axis = 1)
X3 = sm.add_constant(X3)
Y1 = Auto[['mpg']]
X3.columns = ['const', 'hp', 'hp2']
X3.head()

# COMMAND ----------

ols = sm.OLS(Y1, X3).fit()

# COMMAND ----------

ols.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **As with simple linear regression, the standard errors generated through bootstrap is larger than the usual quadratic
# MAGIC regression, because it does not hold any assumptions that regressions need to calculate standard errors.**