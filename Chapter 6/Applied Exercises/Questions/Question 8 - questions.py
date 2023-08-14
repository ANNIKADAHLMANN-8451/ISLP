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

# TODO: your response here

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

# TODO: your response here

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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Repeat (c), using forward stepwise selection and also using backwards stepwise selection. How does your answer compare to the
# MAGIC results in (c)?**

# COMMAND ----------

# TODO: your response here

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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Now generate a response vector Y according to the model**
# MAGIC <br>
# MAGIC <br>
# MAGIC Y = β0 + β7X7 + ",
# MAGIC and perform best subset selection and the lasso. Discuss the
# MAGIC results obtained.

# COMMAND ----------

# TODO: your response here