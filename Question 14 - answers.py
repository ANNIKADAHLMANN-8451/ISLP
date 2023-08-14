# Databricks notebook source
# MAGIC %md
# MAGIC This problem focuses on the collinearity problem.

# COMMAND ----------

# general imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# create data
np.random.seed(1)
x1 = pd.DataFrame(np.array([np.random.uniform(size = 100)]))
x2 = pd.DataFrame(0.5*x1+np.random.standard_normal(size = 100)/10)
y = pd.DataFrame(2+2*x1+0.3*x2+np.random.standard_normal(size = 100))

# COMMAND ----------

x1T= x1.T
x1T.columns = ['x1']
x2T = x2.T
x2T.columns = ['x2']
yT = y.T
yT.columns = ['y']

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Write out the form of the linear model. What are the regression coefcients?**

# COMMAND ----------

# MAGIC %md
# MAGIC y = 2 + 2*x1 + 0.3*x2 + epsilon~N(0,1)
# MAGIC <br>
# MAGIC Beta_0 = 2
# MAGIC <br>
# MAGIC Beta_1 = 2
# MAGIC <br>
# MAGIC Beta_2 = 0.3

# COMMAND ----------

corrx1x2 = np.corrcoef(x1, x2)[0, 1]
print("Correlation between x1 and x2 is: %f\n" % corrx1x2)

# COMMAND ----------

# MAGIC %md
# MAGIC **b. What is the correlation between x1 and x2? Create a scatterplot displaying the relationship between the variables.**

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
plt.scatter(x1, x2, color = 'g')
plt.title("x1 and x2 correlation")
plt.xlabel("x1")
plt.ylabel("y1")

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Using this data, ft a least squares regression to predict `y` using `x1` and `x2`. Describe the results obtained. What are βˆ0, βˆ1, and βˆ2? How do these relate to the true β0, β1, and β2? Can you reject the null hypothesis H0 : β1 = 0? How about the null hypothesis H0 : β2 = 0?**

# COMMAND ----------

X1T = sm.add_constant(x1T)
data = pd.concat([x1T, x2T, yT], axis = 1)
reg_1 = ols("yT~x1T+x2T", data = data).fit()

# COMMAND ----------

reg_1.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Beta_hat_0 = 2.1893
# MAGIC <br>
# MAGIC Beta_hat_1 = 0.7046
# MAGIC <br>
# MAGIC Beta_hat_2 = 2.5024
# MAGIC <br>
# MAGIC Since the p-value of Beta_1 is large, we will reject the null-hypothesis that H_0:Beta_1 = 0. Since the the p-value of Beta_2 is small, we will fail to reject the null-hypothesis that H_0:Beta_2 = 0. However, we could reject it too given the p-value is not too small either.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Now ft a least squares regression to predict `y` using only `x1`. Comment on your results. Can you reject the null hypothesis H0 : β1 = 0?**

# COMMAND ----------

reg_2 = ols("yT~x1T", data = data).fit()

# COMMAND ----------

reg_2.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Beta_hat_0 = 2.2486
# MAGIC <br>
# MAGIC Beta_hat_1 = 1.8770
# MAGIC <br>
# MAGIC Since the p-value of Beta_1 is ~ 0, we will fail to reject the null-hypothesis that H_0:Beta_1 = 0. This is not surprising given the high correlation between
# MAGIC x1 and x2. Multicollinearity supresses the individual effects of each independent variable.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now ft a least squares regression to predict `y` using only `x2`. Comment on your results. Can you reject the null hypothesis H0 : β1 = 0?**

# COMMAND ----------

reg_3 = ols("yT~x2T", data = data).fit()

# COMMAND ----------

reg_3.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Beta_hat_0 = 2.2655
# MAGIC <br>
# MAGIC Beta_hat_1 = 3.5613
# MAGIC <br>
# MAGIC Since the p-value of Beta_2 is ~ 0, we will fail to reject the null-hypothesis that H_0:Beta_12= 0. This is not surprising given the high correlation between
# MAGIC x1 and x2. Multicollinearity supresses the individual effects of each independent variable.

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Do the results obtained in (c)–(e) contradict each other? Explain your answer.**

# COMMAND ----------

# MAGIC %md
# MAGIC No. It is not contradictory. This is due to collinearity as explained above.

# COMMAND ----------

# MAGIC  %md
# MAGIC  **g. Suppose we obtain one additional observation, which was unfortunately mismeasured. We use the function `np.concatenate()` and this additional observation to each of `x1`, `x2` and `y`, as seen below. Re-fit the linear models from (c) to (e) using this new data. What efect does this new observation have on the each of the models? In each model, is this observation an outlier? A high-leverage point? Both? Explain your answers.**

# COMMAND ----------

# additional observation
add_values = pd.DataFrame([0.1, 0.8, 6.0]).T
add_values.columns = ['x1','x2','y']
data = data.append(add_values, ignore_index = False)

# COMMAND ----------

reg_4 = ols("y~x1+x2", data = data).fit()

# COMMAND ----------

reg_4.summary()

# COMMAND ----------

reg_5 = ols("y~x1", data = data).fit()

# COMMAND ----------

reg_5.summary()

# COMMAND ----------

reg_6 = ols("y~x2", data = data).fit()

# COMMAND ----------

reg_6.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Adding the extra row certainly improves the significance of the second independent variable.

# COMMAND ----------

# regression #4: residuals vs fitted plot
plt.xkcd()
fitted_4 = pd.DataFrame([reg_4.predict()]).T
fitted_4.columns = ['fitted']
residuals_4 = reg_4.resid
data_4 = pd.concat([fitted_4, residuals_4.reset_index(drop=True)], axis = 1)
data_4.columns = ['fitted','residuals']
plt.figure(figsize = (25, 10))
reg_plot_4 = sns.regplot(fitted_4, residuals_4, data = data_4)
regline_4 = reg_plot_4.get_lines()[0]
regline_4.set_color('green')
plt.title("Regression#4: Residuals vs Fitted")
plt.xlabel("fitted")
plt.ylabel("residuals")

# COMMAND ----------

# regression #4: normal q-q plot
plt.figure(figsize = (15, 10))
sm.qqplot(reg_4.resid_pearson, fit = True, line = 'r') # another way to do it
plt.title("Regression#4: Normal Q-Q Plot")
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

# regression #4: scale-location plot
plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot(fitted_4, np.sqrt(np.abs(reg_4.resid_pearson)), fit_reg = True, color = 'y')
plt.title("Regression#4: Scale-Location Plot")

# COMMAND ----------

# regression #4: residuals vs leverage plot
plt.xkcd()
fig = plt.figure(figsize = (25, 10))
fig.set_size_inches(30, fig.get_figheight(), forward=True)
sm.graphics.influence_plot(reg_4, criterion="cooks", size = 0.0002**2)
plt.title("Regression#4: Residuals vs Leverage")
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

# regression #5: residuals vs fitted plot
plt.xkcd()
fitted_5 = pd.DataFrame([reg_5.predict()]).T
fitted_5.columns = ['fitted']
residuals_5 = reg_5.resid
data_5 = pd.concat([fitted_5, residuals_5.reset_index(drop=True)], axis = 1)
data_5.columns = ['fitted','residuals']
plt.figure(figsize = (25, 10))
reg_plot_5 = sns.regplot(fitted_5, residuals_5, data = data_5)
regline_5 = reg_plot_5.get_lines()[0]
regline_5.set_color('green')
plt.title("Regression#5: Residuals vs Fitted")
plt.xlabel("fitted")
plt.ylabel("residuals")

# COMMAND ----------

# regression #5: normal q-q plot
plt.figure(figsize = (25, 10))
sm.qqplot(reg_5.resid_pearson, fit = True, line = 'r') # another way to do it
plt.title("Regression#5: Normal Q-Q Plot")
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

# regression #5: scale-location plot
plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot(fitted_5, np.sqrt(np.abs(reg_5.resid_pearson)), fit_reg = True, color = 'y')
plt.title("Regression#5: Scale-Location Plot")

# COMMAND ----------

# regression #5: residuals vs leverage plot
plt.xkcd()
fig = plt.figure(figsize = (25, 10))
fig.set_size_inches(30, fig.get_figheight(), forward=True)
sm.graphics.influence_plot(reg_5, criterion="cooks", size = 0.0002**2)
plt.title("Regression#5: Residuals vs Leverage")
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

# regression #6: residuals vs fitted plot
plt.xkcd()
fitted_6 = pd.DataFrame([reg_6.predict()]).T
fitted_6.columns = ['fitted']
residuals_6 = reg_6.resid
data_6 = pd.concat([fitted_6, residuals_6.reset_index(drop=True)], axis = 1)
data_6.columns = ['fitted','residuals']
plt.figure(figsize = (25, 10))
reg_plot_6 = sns.regplot(fitted_6, residuals_6, data = data_6)
regline_6 = reg_plot_6.get_lines()[0]
regline_6.set_color('green')
plt.title("Regression#6: Residuals vs Fitted")
plt.xlabel("fitted")
plt.ylabel("residuals")

# COMMAND ----------

# regression #5: normal q-q plot
plt.figure(figsize = (25, 10))
sm.qqplot(reg_6.resid_pearson, fit = True, line = 'r') # another way to do it
plt.title("Regression#6: Normal Q-Q Plot")
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

# regression #5: scale-location plot
plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot(fitted_6, np.sqrt(np.abs(reg_5.resid_pearson)), fit_reg = True, color = 'y')
plt.title("Regression#6: Scale-Location Plot")

# COMMAND ----------

# regression #5: residuals vs leverage plot
plt.xkcd()
fig = plt.figure(figsize = (25, 10))
fig.set_size_inches(30, fig.get_figheight(), forward=True)
sm.graphics.influence_plot(reg_6, criterion="cooks", size = 0.0002**2)
plt.title("Regression#6: Residuals vs Leverage")
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC In the first and third model, point 0 is high leverage. The second model does not have any high leverage point, but points 51 and 78 are outliers.
# MAGIC Likewise, in the third model, points 82, 51, 78 are outliers.