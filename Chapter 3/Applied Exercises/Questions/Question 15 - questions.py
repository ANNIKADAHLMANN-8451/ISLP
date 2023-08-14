# Databricks notebook source
$pip install termcolor

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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a multiple regression model to predict the response using all of the predictors. Describe your results. For which predictors can we reject the null hypothesis H0 : βj = 0?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. How do your results from (a) compare to your results from (b)? Create a plot displaying the univariate regression coefcients from (a) on the x-axis, and the multiple regression coefcients from (b) on the y-axis. That is, each predictor is displayed as a single point in the plot. Its coefcient in a simple linear regression model is shown on the x-axis, and its coefcient estimate in the multiple linear regression model is shown on the y-axis.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Is there evidence of non-linear association between any of the predictors and the response? To answer this question, for each predictor X, ft a model of the form**: 
# MAGIC Y = β0 + β1X + β2X2 + β3X3 + e.

# COMMAND ----------

# TODO: your response here