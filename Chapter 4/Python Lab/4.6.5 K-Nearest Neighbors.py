# Databricks notebook source
import numpy as np
import pandas as pd
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

Smarket.head()

# COMMAND ----------

Smarket.info()

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X = np.array(Smarket[['Lag1', 'Lag2']])
y = np.array(Smarket['Direction'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2016, random_state=101)

# COMMAND ----------

# MAGIC %md
# MAGIC **K-Means without standardisation (K = 1)**

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn_1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

# COMMAND ----------

knn_1_pred = knn_1.predict(X_test)

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix

# COMMAND ----------

print(confusion_matrix(y_test, knn_1_pred))

# COMMAND ----------

print(classification_report(y_test, knn_1_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC **K-Means without standardisation (K = 3)**

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn_3 = KNeighborsClassifier().fit(X_train, y_train)

# COMMAND ----------

knn_3_pred = knn_3.predict(X_test)

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix

# COMMAND ----------

print(confusion_matrix(y_test, knn_3_pred))

# COMMAND ----------

print(classification_report(y_test, knn_3_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC *As we can see, increase the number of K marginally improves the precision of the model.*

# COMMAND ----------

# MAGIC %md
# MAGIC **K-Means with standardisation (K = 1)**
# MAGIC <br><br>
# MAGIC **Why standardise?** *Because KNN classifier classifies variables of different sizes, in which distances may vary on an 
# MAGIC absolute scale (e.g. we might be classifying a variable based on house prices (where the distances could be in '000s of 
# MAGIC  £ and age, where the distances could be a few years). Standardisation ensures that these distances are accounted for 
# MAGIC and there "standardised".*

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scaler_1 = StandardScaler()

# COMMAND ----------

scaler_1.fit(Smarket.drop(columns = 'Direction', axis = 1).astype(float))

# COMMAND ----------

scaled_features_1 = scaler_1.transform(Smarket.drop(columns = 'Direction', axis = 1).astype(float))

# COMMAND ----------

df_1 = pd.DataFrame(scaled_features_1, columns = Smarket.columns[:-1] )

# COMMAND ----------

df_1.head()

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(scaled_features_1,Smarket['Direction'],
                                                    test_size=0.30)

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn_s_1 = KNeighborsClassifier(n_neighbors=1)

# COMMAND ----------

knn_s_1.fit(X_train, y_train)

# COMMAND ----------

knn_s_1_pred = knn_s_1.predict(X_test)

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix

# COMMAND ----------

print(confusion_matrix(y_test, knn_s_1_pred))

# COMMAND ----------

print(classification_report(y_test, knn_s_1_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC **K-Means with standardisation (K = 3)**

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scaler_3 = StandardScaler()

# COMMAND ----------

scaler_3.fit(Smarket.drop(columns='Direction', axis = 1).astype(float))

# COMMAND ----------

scaled_features_3 = scaler_3.transform(Smarket.drop(columns='Direction', axis = 1).astype(float))

# COMMAND ----------

df_3 = pd.DataFrame(scaled_features_3, columns = Smarket.columns[:-1] )

# COMMAND ----------

df_3.head()

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(scaled_features_3,Smarket['Direction'],
                                                    test_size=0.30)

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn_s_3 = KNeighborsClassifier(n_neighbors=3)

# COMMAND ----------

knn_s_3.fit(X_train, y_train)

# COMMAND ----------

knn_s_3_pred = knn_s_3.predict(X_test)

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix

# COMMAND ----------

print(confusion_matrix(y_test, knn_s_3_pred))

# COMMAND ----------

print(classification_report(y_test, knn_s_3_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC **As we can see, there is a significant improvement in results with standardisation (precision rate of 85% in models with
# MAGIC standardisation as opposed to 47%-48% in models without standardisation).**