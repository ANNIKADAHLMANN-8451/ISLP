# Databricks notebook source
# import relevant statistical packages
import numpy as np
import pandas as pd

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

list(Smarket)

# COMMAND ----------

Smarket.info()

# COMMAND ----------

Smarket.shape

# COMMAND ----------

Smarket.describe()

# COMMAND ----------

Smarket.corr()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25,10))
plt.plot(Smarket['Volume'], color = 'g')
plt.title("Smarket: Volume")
plt.xlabel("Index")
plt.ylabel("Volume")

# COMMAND ----------

