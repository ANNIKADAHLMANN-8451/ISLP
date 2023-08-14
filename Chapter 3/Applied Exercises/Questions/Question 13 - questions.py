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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Using the `normal()` method, create a vector, `eps`, containing 100 observations drawn from a N(0, 0.25) distribution —a normal distribution with mean zero and variance 0.25.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Using `x` and `eps`, generate a vector `y` according to the model**
# MAGIC <br>
# MAGIC <br>
# MAGIC Y = −1+0.5X + ". (3.39)
# MAGIC
# MAGIC **What is the length of the vector `y`? What are the values of β0 and β1 in this linear model?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Create a scatterplot displaying the relationship between `x` and `y`. Comment on what you observe.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Fit a least squares linear model to predict `y` using `x`. Comment on the model obtained. How do βˆ0 and βˆ1 compare to β0 and β1?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Display the least squares line on the scatterplot obtained in (d). Draw the population regression line on the plot, in a diferent color. Use the `plt.legend()` method of the axes to create an appropriate legend.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Now fit a polynomial regression model that predicts `y` using `x` and `x^2`. Is there evidence that the quadratic term improves the model fit? Explain your answer.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Repeat (a)–(f) after modifying the data generation process in such a way that there is less noise in the data. The model should remain the same. You can do this by decreasing the variance of the normal distribution used to generate the error term _eps_ in (b). Describe your results.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **i. Repeat (a)–(f) after modifying the data generation process in such a way that there is more noise in the data. The model should remain the same. You can do this by increasing the variance of the normal distribution used to generate the error term _eps_ in (b). Describe your results.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **j. What are the confdence intervals for β0 and β1 based on the original data set, the noisier data set, and the less noisy data set? Comment on your results.**

# COMMAND ----------

# TODO: your response here