# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

Smarket.head()

# COMMAND ----------

X = np.array(Smarket[['Lag1', 'Lag2']])
y = np.array(Smarket['Direction'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2016, random_state=101)

# COMMAND ----------

qdafit = QDA().fit(X_train, y_train)

# COMMAND ----------

qdafit.priors_

# COMMAND ----------

qdafit.means_

# COMMAND ----------

qdapred = qdafit.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, qdapred))

# COMMAND ----------

print(classification_report(y_test, qdapred))

# COMMAND ----------

# MAGIC %md
# MAGIC **In this case, the prediction actually reduces. Although one needs to perform further analysis, this suggests that the QDA model actually introduces greater variance into the model which is not compensated by the reduction in bias, thereby 
# MAGIC leading to a poorer fit (the reverse happens in the book, but that is because the method of selection of training and test data in the book and in this notebook are different.**