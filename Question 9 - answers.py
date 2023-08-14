# Databricks notebook source
# MAGIC %md
# MAGIC This question involves the use of multiple linear regression on the `Auto` data set.

# COMMAND ----------

# general imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# COMMAND ----------

# import data visualisation tools
import seaborn as sns

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

int_cols = ["cylinders", "horsepower", "weight", "year", "origin"]
float_cols = ["mpg", "displacement", "acceleration"]
str_cols = ["name"]
Auto[int_cols] = Auto[int_cols].astype(int)
Auto[float_cols] = Auto[float_cols].astype(float)
Auto[str_cols] = Auto[str_cols].astype(str)

# COMMAND ----------

Auto.head()

# COMMAND ----------

list(Auto)

# COMMAND ----------

Auto = Auto.drop(Auto.index[[32, 126, 330, 336, 354]]) # removing rows containing "?". This is the easy way out. Such missing values need to be explored first in a real life situation.

# COMMAND ----------

Auto.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Produce a scatterplot matrix which includes all of the variables in the data set.**

# COMMAND ----------

sns.pairplot(Auto, hue = "origin")

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Compute the matrix of correlations between the variables using the `DataFrame.corr()` method.**

# COMMAND ----------

Auto.corr()

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Use the `sm.OLS()` function to perform a multiple linear regression with `mpg` as the response and all other variables except name as the predictors. Use the `.summary()` function to print the results. Comment on the output. For instance:**
# MAGIC   - i. Is there a relationship between the predictors and the response? Use the `anova_lm()` function from `statsmodels` to answer this question.
# MAGIC   - ii. Which predictors appear to have a statistically signifcant relationship to the response?
# MAGIC   - iii. What does the coefcient for the year variable suggest?

# COMMAND ----------

X = Auto[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'year', 'origin']]
Y = Auto['mpg']
X1 = sm.add_constant(X)
reg = sm.OLS(Y, X1).fit()

# COMMAND ----------

reg.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **e and f. Fit some models with interactions as described in the lab. Do any interactions appear to be statistically signifcant? Try a few different transformations of the variables, such as log, square root, raising to powers. Comment on your findings.**

# COMMAND ----------

X1 = Auto['horsepower']
X2 = Auto['weight']
X3 = Auto['acceleration']
X4 = Auto['year']
X5 = Auto['origin']
X6 = Auto['displacement']
X7 = Auto['cylinders']
Y = Auto['mpg']
reg = ols("Y~X1+X2+X3+X4+X5+X6+X7+X7*X6+X7*X2+X6*X2", data = Auto).fit()

# COMMAND ----------

# MAGIC %md
# MAGIC There is no pure statistical method to assess interaction terms. ISLR  provides some clever examples to deduce this. But ultimately, it will depend
# MAGIC on chopping and changing between different independent variables depending on your research goals. This question on Stats Stack Exchange provides an
# MAGIC excellent answer - http://bit.ly/2ApTvQ4
# MAGIC <br><br>
# MAGIC For the sake of brevity, I have included interactions between terms with higest covariance amongst them.

# COMMAND ----------

reg.summary()

# COMMAND ----------

reg = ols("Y~X1+X2+X3+X4+X5+X6+X7+I(np.log(X1))+I(X4^2)", data = Auto).fit()

# COMMAND ----------

# MAGIC %md
# MAGIC I randomly chose two transformations for two variables:
# MAGIC <br>
# MAGIC 1. Log-transformation for X1: OLS result suggests that for a unit change in log(X1), the miles per gallon reduces by ~27.2 units
# MAGIC <br>
# MAGIC 2. Square of X4: OLS result suggests that for a unit increase in X4^2, the miles per gallon reduces by 0.12 units. However, the high p-value of this statistic suggests that the null hypothesis cannot be rejected. Therefore, essentially there is no difference between this particular value and 0, and therefore this statistic can be discarded.

# COMMAND ----------

reg.summary()