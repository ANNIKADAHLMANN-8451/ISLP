# Databricks notebook source
# MAGIC %md
# MAGIC In this problem we will investigate the t-statistic for the null hypothesis `H0 : β = 0` in simple linear regression without an intercept. To begin, we generate a predictor `x` and a response `y` as follows:

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

# set seed
np.random.seed(1)
x= pd.DataFrame(np.random.normal(0, 1, 100))
y = 2*x + pd.DataFrame(np.random.normal(0, 1, 100))

# COMMAND ----------

data = pd.concat([x,y], axis = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Perform a simple linear regression of `y` onto `x`, without an intercept. Report the coefcient estimate βˆ, the standard error of this coefcient estimate, and the t-statistic and p-value associated with the null hypothesis H0 : β = 0. Comment on these results. (You can perform regression without an intercept using the keywords argument `intercept=False` to `ModelSpec()`.)**

# COMMAND ----------

reg_1 = ols("y~x+0", data = data).fit()

# COMMAND ----------

reg_1.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Now perform a simple linear regression of x onto y without an intercept, and report the coefcient estimate, its standard error, and the corresponding t-statistic and p-values associated with the null hypothesis H0 : β = 0. Comment on these results.**

# COMMAND ----------

reg_2 = ols("x~y+0", data = data).fit()

# COMMAND ----------

reg_2.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Show that when regression is performed with an intercept, the t-statistic for H0 : β1 = 0 is the same for the regression of `y` onto `x` as it is for the regression of `x` onto `y`.**

# COMMAND ----------

reg_3 = ols("y~x", data = data).fit()

# COMMAND ----------

reg_3.summary()

# COMMAND ----------

reg_4 = ols("x~y", data = data).fit()

# COMMAND ----------

reg_4.summary()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot(x[0], y[0], data = data)
plt.title("y~x")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot(y[0], x[0], data = data)
plt.title("x~y")
plt.xlabel("y")
plt.ylabel("x")
plt.show()