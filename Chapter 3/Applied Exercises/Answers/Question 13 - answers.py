# Databricks notebook source
# MAGIC %md
# MAGIC In this exercise you will create some simulated data and will fit simple linear regression models to it. Make sure to use the default random number generator with seed set to 1 prior to starting part (a) to ensure consistent results.

# COMMAND ----------

# general imports
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Using the `standard_normal()` method of your random number generator, create a vector, `x`, containing 100 observations drawn from a N(0, 1) distribution. This represents a feature, X.**

# COMMAND ----------

x = np.random.standard_normal(100)
plt.figure(figsize = (25, 10))
sns.distplot(x, color = 'g')
plt.title("x ~ N(0,1)")

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Using the `normal()` method, create a vector, `eps`, containing 100 observations drawn from a N(0, 0.25) distribution —a normal distribution with mean zero and variance 0.25.**

# COMMAND ----------

eps = np.random.normal(0, 0.25, 100)
plt.figure(figsize = (25, 10))
sns.distplot(eps, color = 'y')
plt.title("eps ~ N(0,0.25)")

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Using `x` and `eps`, generate a vector `y` according to the model**
# MAGIC <br>
# MAGIC <br>
# MAGIC Y = −1+0.5X + ". (3.39)
# MAGIC
# MAGIC **What is the length of the vector `y`? What are the values of β0 and β1 in this linear model?**

# COMMAND ----------

y = -1 + (0.5*x) + eps
print("Length of vector y: ", len(y))
print("Beta_0: ", -1.0)
print("Beta_1: ", 0.5)

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Create a scatterplot displaying the relationship between `x` and `y`. Comment on what you observe.**

# COMMAND ----------

plt.figure(figsize = (25, 10))
plt.scatter(y, x)
plt.title("xy-scatterplot")
plt.xlabel("x")
plt.ylabel("y")

# COMMAND ----------

# MAGIC %md
# MAGIC Observation: linear relationship between x and y with a positive slope.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Fit a least squares linear model to predict `y` using `x`. Comment on the model obtained. How do βˆ0 and βˆ1 compare to β0 and β1?**

# COMMAND ----------

data = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis = 1)
data.columns = ['x', 'y']
reg_1 = ols("y~x", data = data).fit()

# COMMAND ----------

reg_1.summary()

# COMMAND ----------

print("Beta_hat_0: %f" % reg_1.params[0])
print("Beta_hat_1: %f" % reg_1.params[1])

# COMMAND ----------

print("Difference in Beta_0: ", -1.0 - reg_1.params[0])
print("Difference in Beta_1: ", 0.5 - reg_1.params[1])

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Display the least squares line on the scatterplot obtained in (d). Draw the population regression line on the plot, in a diferent color. Use the `plt.legend()` method of the axes to create an appropriate legend.**

# COMMAND ----------

plt.figure(figsize = (25, 10))
reg_plot = sns.regplot(pd.Series(x), pd.Series(y), data = data)
regline = reg_plot.get_lines()[0]
regline.set_color('green')
plt.title("xy-regression_plot")
plt.xlabel("x")
plt.ylabel("y")
labels = ['x', 'y']
plt.legend(labels)

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Now fit a polynomial regression model that predicts `y` using `x` and `x^2`. Is there evidence that the quadratic term improves the model fit? Explain your answer.**

# COMMAND ----------

reg_2 = ols("y~x+I(pow(x, 2))", data = data).fit()
f2 = np.poly1d(np.polyfit(x, y, 2))
print("y = ", f2)

# COMMAND ----------

reg_2.summary()

# COMMAND ----------

RMSE_1 = np.sqrt(reg_1.mse_model) # root mean squared error of the first regression model
RMSE_2 = np.sqrt(reg_2.mse_model) # root mean squared error of the first regression model
print("RMSE_1:", RMSE_1) # this value in the range ~ 5.0
print("RMSE_2:", RMSE_2) # this value in the range of ~ 3.5

# COMMAND ----------

# MAGIC %md
# MAGIC Observation: There is not much to choose between the two models given their R^2. Given that
# MAGIC RMSE_2 is lower than RMSE_1 in general, this suggests the polynomial
# MAGIC model fits the data better. This is because it is able to fit the non-linear
# MAGIC nature of the true model better.

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Repeat (a)–(f) after modifying the data generation process in such a way that there is less noise in the data. The model should remain the same. You can do this by decreasing the variance of the normal distribution used to generate the error term _eps_ in (b). Describe your results.**

# COMMAND ----------

eps_r = eps/4 # the _r connotes "reduced"
y_r = -1 + (0.5*x) + eps_r

# COMMAND ----------

print("Variance of eps: ", eps.var())
print("Variance of eps_r: ", eps_r.var()) # confirms that the overall variance is reduced

# COMMAND ----------

plt.figure(figsize = (25, 10))
plt.scatter(y_r, x)
plt.title("xy_r-scatterplot")
plt.xlabel("x")
plt.ylabel("y_r")

# COMMAND ----------

data_r = pd.concat([pd.DataFrame(x), pd.DataFrame(y_r)], axis = 1)
data.columns = ['x', 'y_r']
reg_r_1 = ols("y_r~x", data = data).fit()
f_r_1 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f_r_1)

# COMMAND ----------

reg_r_1.summary()

# COMMAND ----------

print("Beta_hat_r_0: %f" % reg_r_1.params[0])
print("Beta_hat_r_1: %f" % reg_r_1.params[1])

# COMMAND ----------

print("Difference in Beta_0: ", -1.0 - reg_r_1.params[0])
print("Difference in Beta_1: ", 0.5 - reg_r_1.params[1])

# COMMAND ----------

plt.figure(figsize = (25, 10))
reg_plot_r = sns.regplot(x, y_r, data = data_r)
regline_r = reg_plot_r.get_lines()[0]
regline_r.set_color('green')
plt.title("xy_r-regression_plot")
plt.xlabel("x")
plt.ylabel("y_r")
labels_r = ['x', 'y_r']
plt.legend(labels_r)

# COMMAND ----------

reg_r_2 = ols("y_r~x+I(pow(x, 2))", data = data_r).fit()
f_r_2 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f_r_2)

# COMMAND ----------

reg_r_2.summary()

# COMMAND ----------

RMSE_r_1 = np.sqrt(reg_r_1.mse_model) # root mean squared error of the first regression model
RMSE_r_2 = np.sqrt(reg_r_2.mse_model) # root mean squared error of the first regression model
print("RMSE_r_1:", RMSE_r_1) # this value in the range ~ 5.0
print("RMSE_r_2:", RMSE_r_2) # this value in the range of ~ 3.5

# COMMAND ----------

# MAGIC %md
# MAGIC Observation: Given that RMSE_r_2 is lower than RMSE_r_1 in general, this suggests the
# MAGIC polynomial model fits the data better. The R_2 of the polynomial model also
# MAGIC significantly larger than the first model. This suggests the polynomial model
# MAGIC better explains the reduced variation in data without compromising on the fit.

# COMMAND ----------

# MAGIC %md
# MAGIC **i. Repeat (a)–(f) after modifying the data generation process in such a way that there is more noise in the data. The model should remain the same. You can do this by increasing the variance of the normal distribution used to generate the error term _eps_ in (b). Describe your results.**

# COMMAND ----------

eps_i = eps*4 # the _i connotes "increased"
y_i = -1 + (0.5*x) + eps_i

# COMMAND ----------

print("Sum of square of eps: ", eps.var())
print("Sum of square of eps_i: ",eps_i.var()) # confirms that the overall variance is reduced

# COMMAND ----------

plt.figure(figsize = (25, 10))
plt.scatter(y_i, x)
plt.title("xy_i-scatterplot")
plt.xlabel("x")
plt.ylabel("y_i")

# COMMAND ----------

data_i = pd.concat([pd.DataFrame(x), pd.DataFrame(y_i)], axis = 1)
data.columns = ['x', 'y_i']
reg_i_1 = ols("y_i~x", data = data).fit()
f_i_1 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f_i_1)

# COMMAND ----------

reg_i_1.summary()

# COMMAND ----------

print("Beta_hat_i_0: %f" % reg_i_1.params[0])
print("Beta_hata_i_1: %f" % reg_i_1.params[1])

# COMMAND ----------

print("Difference in Beta_0: ", -1.0 - reg_i_1.params[0])
print("Difference in Beta_1: ", 0.5 - reg_i_1.params[1])

# COMMAND ----------

plt.figure(figsize = (25, 10))
reg_plot_i = sns.regplot(x, y_i, data = data_i)
regline_i = reg_plot_i.get_lines()[0]
regline_i.set_color('green')
plt.title("xy_i-regression_plot")
plt.xlabel("x")
plt.ylabel("y_i")
labels_i = ['x', 'y_i']
plt.legend(labels_i)

# COMMAND ----------

reg_i_2 = ols("y_i~x+I(pow(x, 2))", data = data_i).fit()
f_i_2 = np.poly1d(np.polyfit(x, y, 1))
print("y = ", f_i_2)

# COMMAND ----------

reg_i_2.summary()

# COMMAND ----------

RMSE_i_1 = np.sqrt(reg_i_1.mse_model) # root mean squared error of the first regression model
RMSE_i_2 = np.sqrt(reg_i_2.mse_model) # root mean squared error of the first regression model
print("RMSE_i_1:", RMSE_i_1) # this value in the range ~ 5.0
print("RMSE_i_2:", RMSE_i_2) # this value in the range of ~ 3.5

# COMMAND ----------

# MAGIC %md
# MAGIC Observation: Given that RMSE_r_2 is lower than RMSE_r_1 in general, this suggests the
# MAGIC polynomial model fits the data better. However, the R^2 of the second model is
# MAGIC significantly lower than the first model, which suggests that the polynomial
# MAGIC model starts to follow the noise thereby showing increased variance.

# COMMAND ----------

# MAGIC %md
# MAGIC **j. What are the confdence intervals for β0 and β1 based on the original data set, the noisier data set, and the less noisy data set? Comment on your results.**

# COMMAND ----------

confint1 = pd.DataFrame(reg_1.conf_int(alpha = 0.05)).T
confint2 = pd.DataFrame(reg_2.conf_int(alpha = 0.05)).T
confint_r_1 = pd.DataFrame(reg_r_1.conf_int(alpha = 0.05)).T
confint_r_2 = pd.DataFrame(reg_r_2.conf_int(alpha = 0.05)).T
confint_i_1 = pd.DataFrame(reg_i_1.conf_int(alpha = 0.05)).T
confint_i_2 = pd.DataFrame(reg_i_2.conf_int(alpha = 0.05)).T

# COMMAND ----------

print("95% C.I. of Linear Model:\n")
confint1

# COMMAND ----------

print("95% C.I. of Polynomial Model:\n")
confint2

# COMMAND ----------

print("95% C.I. of Linear Model with reduced variance:\n")
confint_r_1

# COMMAND ----------

print("95% C.I. of Polynomial Model with reduced variance:\n")
confint_r_2

# COMMAND ----------

print("95% C.I. of Polynomial Model with reduced variance:\n")
confint_r_1

# COMMAND ----------

print("95% C.I. of Polynomial Model with increased variance:\n")
confint_r_2

# COMMAND ----------

