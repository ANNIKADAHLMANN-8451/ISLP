# Databricks notebook source
# MAGIC %md
# MAGIC This question should be answered using the `Carseats` data set.

# COMMAND ----------

# import statistical tools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# load data; visualisation same as Section 3.6.3
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Carseats.csv"
CarSeats = spark.read.option("header", "true").csv(url).toPandas()
CarSeats.set_index('SlNo', inplace=True)

str_cols = ["ShelveLoc", "Urban", "US"]
num_cols = ["Sales", "CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]
CarSeats[str_cols] = CarSeats[str_cols].astype(str)
CarSeats[num_cols] = CarSeats[num_cols].astype(float)

# COMMAND ----------

CarSeats.head()

# COMMAND ----------

list(CarSeats)

# COMMAND ----------

CarSeats.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Fit a multiple regression model to predict `Sales` using `Price`, `Urban`, and `US`.**

# COMMAND ----------

reg = ols(formula = 'Sales ~ Price + C(Urban) + C(US)', data = CarSeats).fit() # C prepares categorical data for regression

# COMMAND ----------

reg.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Provide an interpretation of each coefcient in the model. Be careful—some of the variables in the model are qualitative!**

# COMMAND ----------

# MAGIC %md
# MAGIC For a unit increase of price ceterus paribus, the sales decrease by 0.0545 units. Likewise, for a unit increase in an urban setting
# MAGIC ceterus paribus the sales decrease by 0.219 units. Likewise, for a location in the US a unit increase of another store ceterus paribus
# MAGIC increases the sales by 1.2006 units.**

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Write out the model in equation form, being careful to handle the qualitative variables properly.**

# COMMAND ----------

# MAGIC %md
# MAGIC Sales = 13.0435 - 0.0545xPrice - 0.0219 + 1.2006 => Sales = 14.8305 - 0.0545xPrice

# COMMAND ----------

# MAGIC %md
# MAGIC **d. For which of the predictors can you reject the null hypothesis?**

# COMMAND ----------

# MAGIC %md
# MAGIC We can reject "Urban" predictor, given it's high p-value(0.936).

# COMMAND ----------

# MAGIC %md
# MAGIC **e. On the basis of your response to the previous question, fit a smaller model that only uses the predictors for which there is evidence of association with the outcome.**

# COMMAND ----------

reg_1 = ols(formula = 'Sales ~ Price + C(US)', data = CarSeats).fit()

# COMMAND ----------

reg_1.summary()

# COMMAND ----------

# run predictions
predictions_1 = pd.DataFrame(reg_1.predict())
residuals_1 = CarSeats['Sales'] - predictions_1[0]

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.distplot(residuals_1) # residuals are normally distributed. Love it!!!
plt.title("Residual Plot")

# COMMAND ----------

reg_2 = ols(formula = 'Sales ~ Price + C(US)', data = CarSeats).fit()

# COMMAND ----------

reg_2.summary()

# COMMAND ----------

predictions_2 = pd.DataFrame(reg_2.predict())
residuals_2 = CarSeats['Sales'] - predictions_2[0]

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.distplot(residuals_2, color = 'green') # residuals are normally distributed. Love it!!!
plt.title("Residual Plot")

# COMMAND ----------

# MAGIC %md
# MAGIC **f. How well do the models in (a) and (e) ft the data?**

# COMMAND ----------

# error calculations
Y = CarSeats['Sales']
Yhat_1 = predictions_1[0]
Yhat_2 = predictions_2[0]

# COMMAND ----------

from sklearn.metrics import mean_absolute_error, mean_squared_error
MAE_1 = mean_absolute_error(Y, Yhat_1)
MSE_1 = mean_squared_error(Y, Yhat_1)
RMSE_1 = np.sqrt(MSE_1)

# COMMAND ----------

print("Model#1 Mean Absolute Error: %f" % MAE_1)
print("Model#1 Mean Squared Error : %f" % MSE_1)
print("Model#1 Root Mean Squared Error: %f" % RMSE_1)

# COMMAND ----------

MAE_2 = mean_absolute_error(Y, Yhat_2)
MSE_2 = mean_squared_error(Y, Yhat_2)
RMSE_2 = np.sqrt(MSE_2)

# COMMAND ----------

print("Model#1 Mean Absolute Error: %f" % MAE_2)
print("Model#1 Mean Squared Error : %f" % MSE_2)
print("Model#1 Root Mean Squared Error: %f" % RMSE_2)

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Using the model from (e), obtain 95 % confdence intervals for the coefcient(s).**

# COMMAND ----------

# MAGIC %md
# MAGIC From the OLS results, these are the 95% confidence intervals:
# MAGIC <br>
# MAGIC Intercept: (11.790, 14.271)
# MAGIC <br>
# MAGIC US: (0.692, 1.708)
# MAGIC <br>
# MAGIC Price: (-0.065, -0.044)

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Is there evidence of outliers or high leverage observations in the model from (e)?**

# COMMAND ----------

# MAGIC %md
# MAGIC Create plots and find evidence of outliers and high leverage observations.

# COMMAND ----------

# residuals vs fitted plot
plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot(Yhat_2, pd.Series(reg_2.resid_pearson), fit_reg = True, color = 'g')
plt.title("Residuals vs Fitted - Residuals_2")

# COMMAND ----------

# normal q-q plot
plt.xkcd()
plt.figure(figsize = (25, 10))
stats.probplot(residuals_2, plot = plt)
plt.title("Normal Q-Q Plot - Residuals_2 - v1")
plt.show()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sm.qqplot(reg_2.resid_pearson, fit = True, line = 'r') # another way to do it
plt.title("Normal Q-Q Plot - Residuals_2 - v2")
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

# scale-location plot
plt.xkcd()
plt.figure(figsize = (25, 10))
reg_2_sqrt = pd.Series(np.sqrt(np.abs(reg_2.resid_pearson)))
sns.regplot(Yhat_2, reg_2_sqrt, fit_reg = True, color = 'y')
plt.title("Scale-Location Plot - Residuals_2")

# COMMAND ----------

# residuals vs leverage plot
plt.xkcd()
fig = plt.figure(figsize = (25, 10))
fig.set_size_inches(30, fig.get_figheight(), forward=True)
sm.graphics.influence_plot(reg_2, criterion="cooks", size = 0.0002**2)
plt.title("Residuals vs Leverage - Residuals_2")
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Yes, there are high leverage points. Point 42 is one such example.

# COMMAND ----------

