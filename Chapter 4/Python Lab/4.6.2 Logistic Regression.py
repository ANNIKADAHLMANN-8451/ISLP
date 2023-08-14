# Databricks notebook source
# import relevant statistical packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# import relevant data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Smarket.csv"
Smarket = spark.read.option("header", "true").csv(url).toPandas()
Smarket.set_index('SlNo', inplace=True)

str_cols = ["Direction"]
num_cols = list(set(Smarket.columns) - set(str_cols))
Smarket[str_cols] = Smarket[str_cols].astype(str)
Smarket[num_cols] = Smarket[num_cols].astype(float)

# COMMAND ----------

# explore data
Smarket.head()

# COMMAND ----------

Smarket.info()

# COMMAND ----------

X = Smarket[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
Y = Smarket['Direction']

# COMMAND ----------

glmfit = LogisticRegression(solver='liblinear').fit(X, Y)

# COMMAND ----------

glmpred = glmfit.predict(X)

# COMMAND ----------

print(confusion_matrix(Y, glmpred))

# COMMAND ----------

print(classification_report(Y, glmpred))

# COMMAND ----------

