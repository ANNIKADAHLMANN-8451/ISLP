# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Caravan.csv"
Caravan = spark.read.option("header", "true").csv(url).toPandas()
Caravan.set_index('_c0', inplace=True)
Caravan.index.names = ['Index']

str_cols = ["Purchase"]
num_cols = list(set(Caravan.columns) - set(str_cols))
Caravan[str_cols] = Caravan[str_cols].astype(str)
Caravan[num_cols] = Caravan[num_cols].astype(float)

# COMMAND ----------

Caravan.head()

# COMMAND ----------

Caravan.info()

# COMMAND ----------

Caravan['Purchase'].value_counts()

# COMMAND ----------

perc_yes = 348/5822
print("%% of Yes's: %f " % perc_yes)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scaler = StandardScaler()
scaler.fit(Caravan.drop(columns='Purchase', axis = 1).astype(float))
scaled_features = scaler.transform(Caravan.drop(columns='Purchase', axis = 1).astype(float))

# COMMAND ----------

df = pd.DataFrame(scaled_features, columns=Caravan.columns[:-1])

# COMMAND ----------

df.head()

# COMMAND ----------

pf = pd.DataFrame()
for i in Caravan.columns[:-1]:
    pf = pf.append([Caravan[i].var()])


plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(pf.reset_index())

# COMMAND ----------

pf2 = pd.DataFrame()
for i in df.columns:
    pf2 = pf2.append([df[i].var()])


plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(pf2.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC **As we can see, the variance is "standardised" at ~1.**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(scaled_features, Caravan.Purchase, test_size=0.171, random_state=42)

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

# COMMAND ----------

knn_pred = knn.predict(X_test)

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

print(confusion_matrix(y_test, knn_pred))

# COMMAND ----------

print(classification_report(y_test, knn_pred))

# COMMAND ----------

knn3 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# COMMAND ----------

knn3 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# COMMAND ----------

knn3 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# COMMAND ----------

knn3 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# COMMAND ----------

knn5 = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

# COMMAND ----------

knn5_pred = knn5.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knn5_pred))

# COMMAND ----------

print(classification_report(y_test, knn5_pred))

# COMMAND ----------

knn7 = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

# COMMAND ----------

knn7_pred = knn7.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knn7_pred))

# COMMAND ----------

print(classification_report(y_test, knn7_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC **Logistic Regression**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

glmfits = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# COMMAND ----------

glmfits_pred = glmfits.predict(X_test)

# COMMAND ----------

glmfits_pred = glmfits.predict(X_test)

# COMMAND ----------

print(classification_report(y_test, glmfits_pred))

# COMMAND ----------

