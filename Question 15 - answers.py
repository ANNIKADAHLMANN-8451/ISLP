# Databricks notebook source
# MAGIC %pip install --quiet termcolor

# COMMAND ----------

# MAGIC %md
# MAGIC This problem involves the `Boston` data set, which we saw in the lab for this chapter. We will now try to predict per capita crime rate using the other variables in this data set. In other words, per capita crime rate is the response, and the other variables are the predictors.

# COMMAND ----------

# general imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

# COMMAND ----------

# import dataset and preprocess
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas().astype(float)
Boston.set_index('SlNo', inplace=True)

# COMMAND ----------

Boston.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. For each predictor, ft a simple linear regression model to predict the response. Describe your results. In which of the models is there a statistically signifcant association between the predictor and the response? Create some plots to back up your assertions.**

# COMMAND ----------

# run simple linear regressions for each independent variable
for t in Boston.columns:
    reg = ols("crim~Boston[t]", data = Boston).fit()
    print(reg.summary())
    print()
    print(colored("="*78, 'green'))
    print()
    plt.xkcd()
    plt.figure(figsize = (25, 10))
    sns.regplot(reg.predict(), reg.resid, data = Boston)
    plt.title(t)
    plt.xlabel(t)
plt.ylabel('crim')

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a multiple regression model to predict the response using all of the predictors. Describe your results. For which predictors can we reject the null hypothesis H0 : βj = 0?**

# COMMAND ----------

list(Boston)

# COMMAND ----------

# run multivariate linear regression for 'crim'
X1 = Boston.iloc[:,[1,2,4,5,6,7,8, 9,10,11,12,13]]
X2 = Boston['chas']
reg = ols("crim~zn+indus+C(chas)+nox+rm+age+dis+rad\
          +tax+ptratio+black+lstat+medv", data = Boston).fit()

# COMMAND ----------

reg.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **c. How do your results from (a) compare to your results from (b)? Create a plot displaying the univariate regression coefcients from (a) on the x-axis, and the multiple regression coefcients from (b) on the y-axis. That is, each predictor is displayed as a single point in the plot. Its coefcient in a simple linear regression model is shown on the x-axis, and its coefcient estimate in the multiple linear regression model is shown on the y-axis.**

# COMMAND ----------

plt.xkcd()
sns.pairplot(Boston)
plt.title("Boston Pairplot")

# COMMAND ----------

Boston_columns = list(Boston)
for t in Boston_columns:
    reg = ols("crim~Boston[t]+I(pow(Boston[t],2)) +\
              I(pow(Boston[t],3))", data = Boston).fit()
    print(reg.summary())
    print()
    print(colored("="*78, 'green'))
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC 15.a., b. & c. There are statistically significant association between the predictor and response for 'dis', 'rad', 
# MAGIC 'black', 'medv' in the multivariate linear regression model.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Is there evidence of non-linear association between any of the predictors and the response? To answer this question, for each predictor X, ft a model of the form**: 
# MAGIC Y = β0 + β1X + β2X2 + β3X3 + e.

# COMMAND ----------

# MAGIC %md
# MAGIC The answer is 'yes' for all but 'black' and 'chas', as seen in above plots.

# COMMAND ----------

