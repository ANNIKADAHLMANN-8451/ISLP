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

x1T = x1.T
x1T.columns = ['x1']
x2T = x2.T
x2T.columns = ['x2']
yT = y.T
yT.columns = ['y']

data = pd.concat([x1T, x2T, yT], axis = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Write out the form of the linear model. What are the regression coefcients?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. What is the correlation between x1 and x2? Create a scatterplot displaying the relationship between the variables.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Using this data, ft a least squares regression to predict `y` using `x1` and `x2`. Describe the results obtained. What are βˆ0, βˆ1, and βˆ2? How do these relate to the true β0, β1, and β2? Can you reject the null hypothesis H0 : β1 = 0? How about the null hypothesis H0 : β2 = 0?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Now ft a least squares regression to predict `y` using only `x1`. Comment on your results. Can you reject the null hypothesis H0 : β1 = 0?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now ft a least squares regression to predict `y` using only `x2`. Comment on your results. Can you reject the null hypothesis H0 : β1 = 0?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Do the results obtained in (c)–(e) contradict each other? Explain your answer.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC  %md
# MAGIC  **g. Suppose we obtain one additional observation, which was unfortunately mismeasured. We use the function `np.concatenate()` and this additional observation to each of `x1`, `x2` and `y`, as seen below. Re-fit the linear models from (c) to (e) using this new data. What efect does this new observation have on the each of the models? In each model, is this observation an outlier? A high-leverage point? Both? Explain your answers.**

# COMMAND ----------

# additional observation
add_values = pd.DataFrame([0.1, 0.8, 6.0]).T
add_values.columns = ['x1','x2','y']
data = data.append(add_values, ignore_index = False)

# COMMAND ----------

# TODO: your response here