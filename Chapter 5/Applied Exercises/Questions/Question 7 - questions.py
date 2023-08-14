# Databricks notebook source
# MAGIC %md
# MAGIC In Sections 5.1.2 and 5.1.3, we saw that the `cross_validate()` function
# MAGIC can be used in order to compute the LOOCV test error estimate.
# MAGIC Alternatively, one could compute those quantities using just `sm.GLM()`
# MAGIC and the `predict()` method of the ftted model within a for loop. You
# MAGIC will now take this approach in order to compute the LOOCV error
# MAGIC for a simple logistic regression model on the `Weekly` data set. Recall
# MAGIC that in the context of classifcation problems, the LOOCV error is
# MAGIC given in (5.4).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd

# COMMAND ----------

# import relevant data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Weekly.csv"
Weekly = spark.read.option("header", "true").csv(url).toPandas()
Weekly.set_index("_c0", inplace=True)

str_cols = ["Direction"]
int_cols = ['Year']
float_cols = list(set(Weekly.columns)-set(str_cols)-set(int_cols))
Weekly[str_cols] = Weekly[str_cols].astype(str)
Weekly[int_cols] = Weekly[int_cols].astype(int)
Weekly[float_cols] = Weekly[float_cols].astype(float)

# COMMAND ----------

Weekly.head()

# COMMAND ----------

Weekly.info()

# COMMAND ----------

dfX = Weekly.drop(columns='Direction')
dfy = Weekly['Direction']

# COMMAND ----------

dfX.head()

# COMMAND ----------

dfy.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Fit a logistic regression model that predicts `Direction` using `Lag1`
# MAGIC and `Lag2`.**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a logistic regression model that predicts `Direction` using `Lag1`
# MAGIC and `Lag2` using all but the frst observation.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Use the model from (b) to predict the direction of the frst observation. You can do this by predicting that the frst observation
# MAGIC will go up if P(`Direction = "Up"|Lag1, Lag2`) > 0.5. Was this
# MAGIC observation correctly classifed?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Write a for loop from i = 1 to i = n, where n is the number of
# MAGIC observations in the data set, that performs each of the following
# MAGIC steps:**
# MAGIC - i. Fit a logistic regression model using all but the ith observation to predict `Direction` using `Lag1` and `Lag2`.
# MAGIC - ii. Compute the posterior probability of the market moving up for the ith observation.
# MAGIC - iii. Use the posterior probability for the ith observation in order to predict whether or not the market moves up.
# MAGIC - iv. Determine whether or not an error was made in predicting the direction for the ith observation. If an error was made, then indicate this as a 1, and otherwise indicate it as a 0.

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Take the average of the n numbers obtained in (d)iv in order to
# MAGIC obtain the LOOCV estimate for the test error. Comment on the
# MAGIC results.**

# COMMAND ----------

# TODO: your response here