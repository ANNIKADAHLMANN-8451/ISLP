# Databricks notebook source
# MAGIC %md
# MAGIC This question should be answered using the Weekly data set, which is part of the ISLP package. This data is similar in nature to the Smarket data from this chapter’s lab, except that it contains 1,089 weekly returns for 21 years, from the beginning of 1990 to the end of 2010.

# COMMAND ----------

# general imports
import numpy as np
import pandas as pd

# COMMAND ----------

# import data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Weekly.csv"
Weekly = spark.read.option("header", "true").csv(url).toPandas()
Weekly.set_index('_c0', inplace=True)

float_cols = ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today"]
int_cols = ['Year']
str_cols = ["Direction"]
Weekly[float_cols] = Weekly[float_cols].astype(float)
Weekly[int_cols] = Weekly[int_cols].astype(int)
Weekly[str_cols] = Weekly[str_cols].astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Produce some numerical and graphical summaries of the `Weekly` data. Do there appear to be any patterns?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Use the full data set to perform a logistic regression with `Direction` as the response and the fve lag variables plus `Volume` as predictors. Use the summary function to print the results. Do any of the predictors appear to be statistically signifcant? If so, which ones?**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Compute the confusion matrix and overall fraction of correct predictions. Explain what the confusion matrix is telling you about the types of mistakes made by logistic regression.**

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Now ft the logistic regression model using a training data period from 1990 to 2008, with `Lag2` as the only predictor. Compute the confusion matrix and the overall fraction of correct predictions for the held out data (that is, the data from 2009 and 2010).**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Repeat (d) using LDA.**

# COMMAND ----------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Repeat (d) using QDA.**

# COMMAND ----------

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Repeat (d) using KNN with K = 1.**

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **i. Which of these methods appears to provide the best results on this data?**

# COMMAND ----------

# TODO: your response here