# Databricks notebook source
# MAGIC %pip install --quiet mlxtend

# COMMAND ----------

# MAGIC %md
# MAGIC  In this exercise, we will generate simulated data, and will then use
# MAGIC this data to perform forward and backward stepwise selection. Note, this notebook parallels the applied exercises from [ISLR](https://hastie.su.domains/ISLR2/ISLRv2_corrected_June_2023.pdf), rather than the Python eqiuvalent.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd

# COMMAND ----------

# import data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a random number generator and use its `normal()` method
# MAGIC to generate a predictor X of length n = 100, as well as a noise
# MAGIC vector $\ep of length n = 100.**

# COMMAND ----------

X = np.random.normal(size=100)
epsilon = np.random.normal(size=100)

# COMMAND ----------

len(X), len(epsilon)

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Generate a response vector Y of length n = 100 according to
# MAGIC the model**
# MAGIC <br>
# MAGIC <br>
# MAGIC Y = β0 + β1X + β2X2 + β3X3 + ",
# MAGIC <br>
# MAGIC where β0, β1, β2, and β3 are constants of your choice.

# COMMAND ----------

BETA0 = 87.15
BETA1 = -2.85
BETA2 = 37.23
BETA3 = 56.0
Y = BETA0 + (BETA1*X) + (BETA2*pow(X, 2)) + (BETA3*pow(X, 3)) + epsilon

# COMMAND ----------

df = pd.DataFrame([X, Y]).T
df.columns = ['X', 'Y']
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Perform best subset selection
# MAGIC in order to choose the best model containing the predictors
# MAGIC X, X2,...,X10. What is the best model obtained according to
# MAGIC Cp, BIC, and adjusted R2? Show some plots to provide evidence
# MAGIC for your answer, and report the coefcients of the best model obtained.**

# COMMAND ----------

# import custom packages
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as r2
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_linear_regression as PLS
from sklearn.metrics import mean_squared_error

# COMMAND ----------

dfX = pd.DataFrame([X, pow(X, 2), pow(X, 3), pow(X, 4), pow(X, 5), pow(X, 6), pow(X, 7), pow(X, 8), pow(X, 9), pow(X, 10)]).T
dfX.columns = ['X', 'X^2', 'X^3', 'X^4', 'X^5', 'X^6', 'X^7', 'X^8', 'X^9', 'X^10']
dfX.head()

# COMMAND ----------

dfY = pd.DataFrame(df['Y'])
dfY.head()

# COMMAND ----------

lm = LinearRegression()
efs = EFS(lm, min_features=1, max_features=10, scoring='r2', cv=10).fit(dfX, dfY)

# COMMAND ----------

SCORE = pd.DataFrame()
for k in efs.get_metric_dict():
    score = efs.get_metric_dict()[k]['std_err']
    SCORE = SCORE.append([score])

SCORE.columns = ['std err']
SCORE.reset_index(drop=True, inplace=True)

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(SCORE)

# COMMAND ----------

minSCORE = SCORE.loc[SCORE['std err'] == min(SCORE['std err'])]
minSCORE

# COMMAND ----------

efs.subsets_[180]

# COMMAND ----------

features = efs.best_feature_names_
idx = efs.best_idx_

feat_list = pd.concat([pd.DataFrame([features]).T, pd.DataFrame([idx]).T], axis = 1)
feat_list.columns = ['Features', 'Index']
avg_score = pd.DataFrame()
for i in idx:
    avg_score = avg_score.append(pd.DataFrame([efs.subsets_[i]['avg_score']]))
avg_score.columns = ['Average Score']
avg_score.reset_index(drop=True,inplace=True)
feat_list = pd.concat([feat_list, avg_score], axis=1, ignore_index=True)
feat_list.columns = ['Features', 'Index', 'Average Score']
feat_list

# COMMAND ----------

# performing linear regression on best features
dfX_reg = dfX[feat_list['Features']]

lm_best = LinearRegression().fit(dfX_reg, dfY)
final_coefs = pd.DataFrame(lm_best.coef_).T
final_coefs = pd.concat([feat_list['Features'], final_coefs], axis = 1)
final_intercept = pd.DataFrame([lm_best.intercept_]*4)
final_coefs = pd.concat([final_coefs, final_intercept], axis = 1)
final_coefs.columns = ['Features', 'Coefficients', 'Intercept']
final_coefs

# COMMAND ----------

lmpred = lm_best.predict(dfX_reg)
print("Mean Squared Error: ", round(mean_squared_error(dfY, lmpred), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the best equation through best subset selection is $Y$ = 87.289744 - 2.628061$X$ + 37.113913$X^2$ + 55.792262$X^3$ + 0.000733$X^9$ and the mean squared error is 0.813.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Repeat (c), using forward stepwise selection and also using backwards stepwise selection. How does your answer compare to the
# MAGIC results in (c)?**

# COMMAND ----------

# MAGIC %md
# MAGIC Forward stepwise selection

# COMMAND ----------

# import custom packages
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# COMMAND ----------

lmf = LinearRegression()
sfs = SFS(lmf, k_features=(1,len(dfX.columns)), forward=True, floating=False, scoring='r2',cv=10).fit(dfX.values, dfY)

# COMMAND ----------

FSCORE = pd.DataFrame()
for k in sfs.get_metric_dict():
    fscore = sfs.get_metric_dict()[k]['avg_score']
    FSCORE = FSCORE.append([fscore])

FSCORE.columns = ['avg score']
FSCORE.reset_index(drop=True, inplace=True)
FSCORE.index = FSCORE.index+1

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(FSCORE)

# COMMAND ----------

FSCORE

# COMMAND ----------

# MAGIC %md
# MAGIC Forward stepwise regression has the higest $R^2$ for n=4.

# COMMAND ----------

dfX.columns

# COMMAND ----------

feat = sfs.k_feature_idx_
feat

# COMMAND ----------

feat_list = pd.DataFrame()
for i in feat:
    print(dfX.columns[i])
    feat_list = feat_list.append(pd.DataFrame([dfX.columns[i]]))
feat_list.reset_index(inplace=True, drop=True)
feat_list.columns = ['Features']
feat_list

# COMMAND ----------

# MAGIC %md
# MAGIC *It's interesting to note that forward stepwise regression returns the same predictors as that of best subset selection based on $R^2$.*

# COMMAND ----------

# performing linear regression on best features
dfX_reg = dfX[feat_list['Features']]

lm_best = LinearRegression().fit(dfX_reg, dfY)
final_coefs = pd.DataFrame(lm_best.coef_).T
final_coefs = pd.concat([feat_list['Features'], final_coefs], axis = 1)
final_intercept = pd.DataFrame([lm_best.intercept_]*4)
final_coefs = pd.concat([final_coefs, final_intercept], axis = 1)
final_coefs.columns = ['Features', 'Coefficients', 'Intercept']
final_coefs

# COMMAND ----------

lmpred = lm_best.predict(dfX_reg)
print("Mean Squared Error: ", round(mean_squared_error(dfY, lmpred), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the best equation through forward stepwise selection is $Y$ = 87.289744 - 2.628061$X$ + 37.113913$X^2$ + 55.792262$X^3$ + 0.000733$X^9$ and the mean squared error is 0.813.

# COMMAND ----------

# MAGIC %md
# MAGIC Backward stepwise selection

# COMMAND ----------

lmb = LinearRegression()
sfs = SFS(lmb, k_features=(1,len(dfX.columns)), forward=False, floating=False, scoring='r2',cv=10).fit(dfX.values, dfY)

# COMMAND ----------

BSCORE = pd.DataFrame()
for k in sfs.get_metric_dict():
    bscore = sfs.get_metric_dict()[k]['avg_score']
    BSCORE = BSCORE.append([bscore])

BSCORE.columns = ['avg score']
BSCORE.reset_index(drop=True, inplace=True)
BSCORE.index = BSCORE.index+1

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(BSCORE)

# COMMAND ----------

BSCORE

# COMMAND ----------

# MAGIC %md
# MAGIC Forward stepwise regression has the higest $R^2$ for n=4.

# COMMAND ----------

dfX.columns

# COMMAND ----------

feat = sfs.k_feature_idx_
feat

# COMMAND ----------

feat_list = pd.DataFrame()
for i in feat:
    print(dfX.columns[i])
    feat_list = feat_list.append(pd.DataFrame([dfX.columns[i]]))
feat_list.reset_index(inplace=True, drop=True)
feat_list.columns = ['Features']
feat_list

# COMMAND ----------

# MAGIC %md
# MAGIC *It's interesting to note that backward stepwise regression returns the same predictors as that of best subset selection  and forward stepwise selection based on $R^2$.*

# COMMAND ----------

# performing linear regression on best features
dfX_reg = dfX[feat_list['Features']]

lm_best = LinearRegression().fit(dfX_reg, dfY)
final_coefs = pd.DataFrame(lm_best.coef_).T
final_coefs = pd.concat([feat_list['Features'], final_coefs], axis = 1)
final_intercept = pd.DataFrame([lm_best.intercept_]*4)
final_coefs = pd.concat([final_coefs, final_intercept], axis = 1)
final_coefs.columns = ['Features', 'Coefficients', 'Intercept']
final_coefs

# COMMAND ----------

lmpred = lm_best.predict(dfX_reg)
print("Mean Squared Error: ", round(mean_squared_error(dfY, lmpred), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the best equation through backward stepwise selection is $Y$ = 87.289744 - 2.628061$X$ + 37.113913$X^2$ + 55.792262$X^3$ + 0.000733$X^9$ and the mean squared error is 0.813.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now ft a lasso model to the simulated data, again using X, X2,
# MAGIC ...,X10 as predictors. Use cross-validation to select the optimal
# MAGIC value of λ. Create plots of the cross-validation error as a function
# MAGIC of λ. Report the resulting coefcient estimates, and discuss the
# MAGIC results obtained.**

# COMMAND ----------

# import custom packages
from sklearn.linear_model import Lasso

# COMMAND ----------

# standardisation
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(dfX)
dfX_scaled = scaler.transform(dfX)

# COMMAND ----------

n = 100
lambdas = (np.logspace(10, -2, num=100))

# COMMAND ----------

coefs = []
MSE = []
for k in lambdas:
    lassomod = Lasso(alpha=k, fit_intercept=True).fit(dfX_scaled,dfY)
    coefs.append(lassomod.coef_)
    MSE.append(mean_squared_error(dfY, lassomod.predict(dfX_scaled)))

# COMMAND ----------

coefPD = pd.DataFrame(coefs)

# COMMAND ----------

coefPD.head()

# COMMAND ----------

coefPD.shape

# COMMAND ----------

lassomod.intercept_

# COMMAND ----------

coefPD = pd.concat([pd.DataFrame([lassomod.intercept_]*100), coefPD], axis=1)
coefPD.columns = ['Intercept', 'X', 'X^2', 'X^3', 'X^4', 'X^5', 'X^6', 'X^7', 'X^8', 'X^9', 'X^10']
coefPD.shape

# COMMAND ----------

coefPD.head()

# COMMAND ----------

plt.xkcd()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,10))

# indexing the five largest coefficients
idx = np.argpartition(np.abs(coefs[-1]), -5)[-5:]

# standardised coefficients vs lambdas
ax1.plot(lambdas, coefs)
ax1.set_xscale('log')
ax1.set_xlabel('lambda')
ax1.set_ylabel('standardized soefficients')
ax1.set_title('standardised coefficients vs lambdas')
ax1.legend(np.array(ax1.get_lines())[idx], dfX.columns[idx])

# standardised coefficients vs l2 norms
l2norm = np.linalg.norm(coefs[-1])
l2coefs = np.linalg.norm(coefs/l2norm, axis=1)
ax2.plot(l2coefs, coefs)
ax2.set_xlabel('l2 norm of ridge coefficients / l2 norm of least squares coefficients')
ax2.set_ylabel('standardized coefficients')
ax2.set_title('standardised coefficients vs l2 norms')
ax2.legend(np.array(ax2.get_lines())[idx], dfX.columns[idx]);

# 'Mean Square Error(MSE) vs lambdas
ax3.plot(lambdas, MSE)
ax3.set_xscale('log')
ax3.set_xlabel('lambda')
ax3.set_ylabel('Mean Square Error(MSE)')
ax3.set_title('Mean Square Error(MSE) vs lambdas');

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.5, random_state=42)

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

# finding the best lambda using CV
from sklearn.linear_model import LassoCV
lmlassoCV = LassoCV(alphas=lambdas, cv=10).fit(X_train_scaled, y_train)
lmlassoCValpha = lmlassoCV.alpha_
print("Best lambda: ", lmlassoCValpha)

# COMMAND ----------

# performing lasso regression using best lambda
lmlasso = Lasso(alpha=lmlassoCValpha, fit_intercept=True).fit(X_train_scaled, y_train)
lmlasso_MSE = mean_squared_error(y_test, lmlasso.predict(X_test_scaled))
print('MSE for best lambda: ', round(lmlasso_MSE, 3))

# COMMAND ----------

intercept_list = pd.DataFrame([lmlasso.intercept_]*10)
coef_list = pd.concat([intercept_list, pd.DataFrame([lmlasso.coef_]).T], axis = 1)
coef_list.reset_index(inplace=True, drop=True)
coef_list.columns = ['Intercept', 'Coefficients']
coef_list.index = coef_list.index+1
coef_list

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the best equation through lasso regression is $Y$ = 100.10319 - 1.621967$X$ +  41.587344$X^2$ + 144.624937$X^3$ + 2.440223$X^4$ + 1.442031$X^5$ - 1.194567$X^8$ - 0.050617$X^{10}$ and the mean squared error is 1.382.

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Now generate a response vector Y according to the model**
# MAGIC <br>
# MAGIC <br>
# MAGIC Y = β0 + β7X7 + ",
# MAGIC and perform best subset selection and the lasso. Discuss the
# MAGIC results obtained.

# COMMAND ----------

BETA7 = 23.457
Y7 = BETA0 + (BETA7*pow(X, 7)) + epsilon

# COMMAND ----------

# MAGIC %md
# MAGIC Best subset selection

# COMMAND ----------

dfX7 = pd.DataFrame([pow(X, 7)]).T
dfX7.columns = ['X^7']
dfY7 = pd.DataFrame([Y7]).T
dfY7.columns = ['Y']

# COMMAND ----------

lm = LinearRegression()
efs = EFS(lm, min_features=1, max_features=1, scoring='r2', cv=10).fit(dfX7, dfY7)

# COMMAND ----------

SCORE = pd.DataFrame()
for k in efs.get_metric_dict():
    score = efs.get_metric_dict()[k]['std_err']
    SCORE = SCORE.append([score])

SCORE.columns = ['std err']
SCORE.reset_index(drop=True, inplace=True)

# COMMAND ----------

minSCORE = SCORE.loc[SCORE['std err'] == min(SCORE['std err'])]
minSCORE

# COMMAND ----------

efs.subsets_[0]

# COMMAND ----------

features = efs.best_feature_names_
idx = efs.best_idx_

feat_list = pd.concat([pd.DataFrame([features]).T, pd.DataFrame([idx]).T], axis = 1)
feat_list.columns = ['Features', 'Index']
avg_score = pd.DataFrame()
for i in idx:
    avg_score = avg_score.append(pd.DataFrame([efs.subsets_[i]['avg_score']]))
avg_score.columns = ['Average Score']
avg_score.reset_index(drop=True,inplace=True)
feat_list = pd.concat([feat_list, avg_score], axis=1, ignore_index=True)
feat_list.columns = ['Features', 'Index', 'Average Score']
feat_list

# COMMAND ----------

# performing linear regression on best features
dfX_reg = dfX[feat_list['Features']]

lm_best = LinearRegression().fit(dfX_reg, dfY)
final_coefs = pd.DataFrame(lm_best.coef_).T
final_coefs = pd.concat([feat_list['Features'], final_coefs], axis = 1)
final_intercept = pd.DataFrame([lm_best.intercept_])
final_coefs = pd.concat([final_coefs, final_intercept], axis = 1)
final_coefs.columns = ['Features', 'Coefficients', 'Intercept']
final_coefs

# COMMAND ----------

lmpred = lm_best.predict(dfX_reg)
print("Mean Squared Error: ", round(mean_squared_error(dfY, lmpred), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the best equation through best subset selection is $Y$ = 2.128803 + 112.78394$X^7$ and the mean squared error is 5782.033.

# COMMAND ----------

# MAGIC %md
# MAGIC Lasso regression

# COMMAND ----------

# standardization
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(dfX7)
dfX7_scaled = scaler.transform(dfX7)

# COMMAND ----------

n = 100
lambdas = (np.logspace(10, -2, num=100))

# COMMAND ----------

coefs = []
MSE = []
for k in lambdas:
    lassomod = Lasso(alpha=k, fit_intercept=True).fit(dfX7_scaled,dfY7)
    coefs.append(lassomod.coef_)
    MSE.append(mean_squared_error(dfY7, lassomod.predict(dfX7_scaled)))

# COMMAND ----------

coefPD = pd.DataFrame(coefs)

# COMMAND ----------

coefPD.head()

# COMMAND ----------

coefPD.shape

# COMMAND ----------

lassomod.intercept_

# COMMAND ----------

coefPD = pd.concat([pd.DataFrame([lassomod.intercept_]*100), coefPD], axis=1)
coefPD.columns = ['Intercept', 'X^7']
coefPD.shape

# COMMAND ----------

coefPD.head()

# COMMAND ----------

plt.xkcd()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,10))

# indexing the only coefficient
idx = np.argpartition(np.abs(coefs[-1]), -1)[-1:]

# standardised coefficients vs lambdas
ax1.plot(lambdas, coefs)
ax1.set_xscale('log')
ax1.set_xlabel('lambda')
ax1.set_ylabel('standardized soefficients')
ax1.set_title('standardised coefficients vs lambdas')
ax1.legend(np.array(ax1.get_lines())[idx], dfX.columns[idx])

# standardised coefficients vs l2 norms
l2norm = linalg.norm(coefs[-1])
l2coefs = linalg.norm(coefs/l2norm, axis=1)
ax2.plot(l2coefs, coefs)
ax2.set_xlabel('l2 norm of ridge coefficients / l2 norm of least squares coefficients')
ax2.set_ylabel('standardized coefficients')
ax2.set_title('standardised coefficients vs l2 norms')
ax2.legend(np.array(ax2.get_lines())[idx], dfX.columns[idx]);

# 'Mean Square Error(MSE) vs lambdas
ax3.plot(lambdas, MSE)
ax3.set_xscale('log')
ax3.set_xlabel('lambda')
ax3.set_ylabel('Mean Square Error(MSE)')
ax3.set_title('Mean Square Error(MSE) vs lambdas');

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dfX7, dfY7, test_size=0.5, random_state=42)

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

# finding the best lambda using CV
from sklearn.linear_model import LassoCV
lmlassoCV = LassoCV(alphas=lambdas, cv=10).fit(X_train_scaled, y_train)
lmlassoCValpha = lmlassoCV.alpha_
print("Best lambda: ", lmlassoCValpha)

# COMMAND ----------

# performing lasso regression using best lambda
lmlasso = Lasso(alpha=lmlassoCValpha, fit_intercept=True).fit(X_train_scaled, y_train)
lmlasso_MSE = mean_squared_error(y_test, lmlasso.predict(X_test_scaled))
print('MSE for best lambda: ', round(lmlasso_MSE, 3))

# COMMAND ----------

intercept_list = pd.DataFrame([lmlasso.intercept_])
coef_list = pd.concat([intercept_list, pd.DataFrame([lmlasso.coef_]).T], axis = 1)
coef_list.reset_index(inplace=True, drop=True)
coef_list.columns = ['Intercept', 'Coefficients']
coef_list.index = coef_list.index+1
coef_list

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the best equation through lasso regression is $Y$ = -72.609319 + 1433.946961$X^7$ and the mean squared error is 0.859.

# COMMAND ----------

# MAGIC %md
# MAGIC This suggests that when p<<n, then lasso provides a better fit (even if the intercept is quite off from its true value)
# MAGIC than best subset selection as suggested by the lower MSE through lasso regression. But, as p  increases to the point of
# MAGIC p$\approx$n, then best subset selection is more preferred, albeit it comes at a cost of greater computational requirement.

# COMMAND ----------

