# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

from sklearn.model_selection import train_test_split

# COMMAND ----------

X = np.array(Smarket[['Lag1', 'Lag2']])
y = np.array(Smarket['Direction'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2016, random_state=101)

# COMMAND ----------

ldafit = LDA()
ldafit.fit(X_train, y_train)

# COMMAND ----------

ldafit.priors_

# COMMAND ----------

ldafit.means_

# COMMAND ----------

ldafit.coef_

# COMMAND ----------

ldapred = ldafit.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, ldapred))

# COMMAND ----------

print(classification_report(y_test, ldapred))

# COMMAND ----------

# MAGIC %md
# MAGIC **The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. Therefore,
# MAGIC this model is able to correctly classify 57% of the test data (which I believe is great given the vagaries of the stock 
# MAGIC market!!**