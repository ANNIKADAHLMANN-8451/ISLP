# Databricks notebook source
# MAGIC %md
# MAGIC We will now perform cross-validation on a simulated data set.

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

np.random.seed(1)

# COMMAND ----------

x = np.random.normal(size=100)

# COMMAND ----------

x

# COMMAND ----------

# MAGIC %md 
# MAGIC **a. In this data set, what is n and what is p? Write out the model
# MAGIC used to generate the data in equation form.**

# COMMAND ----------

len(x)

# COMMAND ----------

x.mean(), np.sqrt(x.var()) # so roughly standard normal

# COMMAND ----------

y = x - 2 * np.power(x, 2) + np.random.normal(size=100)

# COMMAND ----------

y

# COMMAND ----------

len(y)

# COMMAND ----------

# MAGIC %md
# MAGIC n = 100
# MAGIC <br>
# MAGIC p = 2
# MAGIC <br>
# MAGIC y = X - 2X$^{2}$ + $\epsilon$

# COMMAND ----------

# MAGIC %md
# MAGIC **b.  Create a scatterplot of X against Y . Comment on what you find.**

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
plt.scatter(x, y, color = 'green')
plt.title("Scatterplot of X against Y")
plt.xlabel("x")
plt.ylabel("y")

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Set a random seed, and then compute the LOOCV errors that
# MAGIC result from ftting the following four models using least squares:**
# MAGIC <br>
# MAGIC <br>
# MAGIC i. Y = β0 + β1X + e
# MAGIC <br>
# MAGIC ii. Y = β0 + β1X + β2X2 + e
# MAGIC <br>
# MAGIC iii. Y = β0 + β1X + β2X2 + β3X3 + e
# MAGIC <br>
# MAGIC iv. Y = β0 + β1X + β2X2 + β3X3 + β4X4 + e.

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

from sklearn.model_selection import LeaveOneOut, train_test_split

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

from sklearn.metrics import confusion_matrix, mean_squared_error

# COMMAND ----------

np.random.seed(1)

# COMMAND ----------

x = x.reshape(-1,1)

# COMMAND ----------

X = pd.DataFrame(x)
Y = pd.DataFrame(y)

# COMMAND ----------

X.columns = ['X']
Y.columns = ['Y']

# COMMAND ----------

X.head()

# COMMAND ----------

Y.head()

# COMMAND ----------

loo = LeaveOneOut()

# COMMAND ----------

total_sets = loo.get_n_splits(X)

# COMMAND ----------

MSE_all_OLS = pd.DataFrame()
MSE_all_LOOCV = pd.DataFrame()

# COMMAND ----------

for i in range(1, 5):
    
    MSE_OLS = 0
    MSE_LOOCV = 0
    
    X = pd.DataFrame(x)
    X_ = pd.DataFrame(PF(i).fit_transform(X))
    X_.drop(columns=0, inplace=True)
    Y = pd.DataFrame(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_, Y, test_size=0.5, random_state=42)
    lmfit = LinearRegression().fit(X_train, y_train)
    lmpred = lmfit.predict(X_test)
    MSE_OLS += mean_squared_error(y_test, lmpred)
    MSE_OLS_mean = MSE_OLS/total_sets
    MSE_all_OLS = MSE_all_OLS.append([MSE_OLS])
    
    for train_index, test_index in loo.split(X):
        X1_train, X1_test = X_.iloc[train_index], X_.iloc[test_index]
        y1_train, y1_test = Y.iloc[train_index], Y.iloc[test_index]
        lmfit1 = LinearRegression().fit(X1_train, y1_train)
        lmpred1 = lmfit1.predict(X1_test)
        MSE_LOOCV += mean_squared_error(y1_test, lmpred1)
    
    MSE_LOOCV_mean = MSE_LOOCV/total_sets
    MSE_all_LOOCV = MSE_all_LOOCV.append([MSE_LOOCV_mean])

# COMMAND ----------

MSE_all_OLS.reset_index(drop=True, inplace=True)

# COMMAND ----------

MSE_all_LOOCV.reset_index(drop=True, inplace=True)

# COMMAND ----------

MSE_all_OLS.columns = ['MSE_OLS']
MSE_all_LOOCV.columns =['MSE_LOOCV']

# COMMAND ----------

MSE_all = pd.concat([MSE_all_OLS, MSE_all_LOOCV], axis = 1)

# COMMAND ----------

MSE_all

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Repeat (c) using another random seed, and report your results.
# MAGIC Are your results the same as what you got in (c)? Why?**

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF

# COMMAND ----------

from sklearn.model_selection import LeaveOneOut, train_test_split

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

from sklearn.metrics import confusion_matrix, mean_squared_error

# COMMAND ----------

np.random.seed(67)

# COMMAND ----------

x = x.reshape(-1,1)

# COMMAND ----------

X = pd.DataFrame(x)
Y = pd.DataFrame(y)

# COMMAND ----------

X.columns = ['X']
Y.columns = ['Y']

# COMMAND ----------

X.head()

# COMMAND ----------

Y.head()

# COMMAND ----------

loo = LeaveOneOut()

# COMMAND ----------

total_sets = loo.get_n_splits(X)

# COMMAND ----------

MSE_all_OLS = pd.DataFrame()
MSE_all_LOOCV = pd.DataFrame()

# COMMAND ----------

for i in range(1, 5):
    
    MSE_OLS = 0
    MSE_LOOCV = 0
    
    X = pd.DataFrame(x)
    X_ = pd.DataFrame(PF(i).fit_transform(X))
    X_.drop(columns=0, inplace=True)
    Y = pd.DataFrame(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_, Y, test_size=0.5, random_state=42)
    lmfit = LinearRegression().fit(X_train, y_train)
    lmpred = lmfit.predict(X_test)
    MSE_OLS += mean_squared_error(y_test, lmpred)
    MSE_OLS_mean = MSE_OLS/total_sets
    MSE_all_OLS = MSE_all_OLS.append([MSE_OLS])
    
    for train_index, test_index in loo.split(X):
        X1_train, X1_test = X_.iloc[train_index], X_.iloc[test_index]
        y1_train, y1_test = Y.iloc[train_index], Y.iloc[test_index]
        lmfit1 = LinearRegression().fit(X1_train, y1_train)
        lmpred1 = lmfit1.predict(X1_test)
        MSE_LOOCV += mean_squared_error(y1_test, lmpred1)
    
    MSE_LOOCV_mean = MSE_LOOCV/total_sets
    MSE_all_LOOCV = MSE_all_LOOCV.append([MSE_LOOCV_mean])

# COMMAND ----------

MSE_all_OLS.reset_index(drop=True, inplace=True)

# COMMAND ----------

MSE_all_LOOCV.reset_index(drop=True, inplace=True)

# COMMAND ----------

MSE_all_OLS.columns = ['MSE_OLS']
MSE_all_LOOCV.columns =['MSE_LOOCV']

# COMMAND ----------

MSE_all = pd.concat([MSE_all_OLS, MSE_all_LOOCV], axis = 1)

# COMMAND ----------

MSE_all

# COMMAND ----------

# MAGIC %md
# MAGIC We get the same result because LOOCV goes through the same n iterations of a single observations. Hence, it is not affected
# MAGIC in any way by the random seed.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Which of the models in (c) had the smallest LOOCV error? Is
# MAGIC this what you expected? Explain your answer.**

# COMMAND ----------

# MAGIC %md
# MAGIC This can be explained by the fact that by increasing the order from linear to qudratic model, we reduce the bias, without significant increase in the variance. However, as we keep increasing the order of polynomials, the variance starts to increase thereby causing an increase in the overall MSE. In other words, the quadratic model most closely
# MAGIC matches the true shape of Y.

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Comment on the statistical signifcance of the coefcient estimates that results from ftting each of the models in (c) using
# MAGIC least squares. Do these results agree with the conclusions drawn
# MAGIC based on the cross-validation results?**

# COMMAND ----------

import statsmodels.api as sm

# COMMAND ----------

for i in range(1, 5):
    X = pd.DataFrame(x)
    X_ = pd.DataFrame(PF(i).fit_transform(X))
    X_.drop(columns=0, inplace=True)
    X_ = sm.add_constant(X_)
    Y = pd.DataFrame(y)
    lmfit = sm.OLS(Y, X_).fit()
    candp = pd.concat([round(lmfit.params, 4), round(lmfit.pvalues, 4)], axis = 1)
    candp.columns = ['Coefficients', 'pvalues']
    print(candp)
    print("\n============================\n")

# COMMAND ----------

# MAGIC %md
# MAGIC In each instance, the linear (apart from the first model) and the quadratic terms are the only statistically significant terms. For the first model, only the constant is statistically significant. This explains the large MSE, since the there is unlikely to be a substantial association between the linear term and the dependent variable due to chance alone. Both linear and quadratic terms are significant in the second model. This suggests that the quadratic model most likely closely explain the shape of the true distribution of the dependent variable (which is reflected by a significant drop in MSE over the first model). The cubic and quartic terms are not significant in the third and fourth model, which suggests that any substantial relationship between the dependent variable and the cubic and quartic terms are likely due to chance alone. This reflects in the relatively minor increase in MSE of the cubic and quartic models.

# COMMAND ----------

