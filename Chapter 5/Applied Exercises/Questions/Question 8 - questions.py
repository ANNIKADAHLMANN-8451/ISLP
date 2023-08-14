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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b.  Create a scatterplot of X against Y . Comment on what you find.**

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# TODO: your response here

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
from sklearn.linear_model import LinearRaegression
from sklearn.model_selection import LeaveOneOut, train_test_split
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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Repeat (c) using another random seed, and report your results.
# MAGIC Are your results the same as what you got in (c)? Why?**

# COMMAND ----------

from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.linear_model import LinearRegression
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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Which of the models in (c) had the smallest LOOCV error? Is
# MAGIC this what you expected? Explain your answer.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Comment on the statistical signifcance of the coefcient estimates that results from ftting each of the models in (c) using
# MAGIC least squares. Do these results agree with the conclusions drawn
# MAGIC based on the cross-validation results?**

# COMMAND ----------

import statsmodels.api as sm

# COMMAND ----------

# TODO: your response here