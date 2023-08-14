# Databricks notebook source
# MAGIC %md
# MAGIC ### Preprocessing

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

# import custom packages
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Hitters.csv"
Hitters = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["Names", "NewLeague", "League", "Division"]
num_cols = list(set(Hitters.columns) - set(str_cols))
Hitters["Salary"] = np.where(Hitters["Salary"] == "NA", np.nan, Hitters["Salary"])
Hitters[str_cols] = Hitters[str_cols].astype(str)
Hitters[num_cols] = Hitters[num_cols].astype(float)

# COMMAND ----------

Hitters.head()

# COMMAND ----------

# clean data
print(Hitters.shape)
Hitters = Hitters.dropna()

# COMMAND ----------

Hitters.shape

# COMMAND ----------

Hitters.head()

# COMMAND ----------

# converting categorical data into dummy variable
Hitters_1 = pd.get_dummies(Hitters, drop_first=True, columns=['League', 'Division', 'NewLeague'])
Hitters_1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split dataset into training and test dataset (and standardise them)

# COMMAND ----------

from sklearn.model_selection import train_test_split
X = Hitters_1.drop(columns = ['Salary', 'Names'])
y = Hitters_1.Salary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Partial least squares regression

# COMMAND ----------

from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.preprocessing import scale

# COMMAND ----------

n = len(X_train)
n

# COMMAND ----------

from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')
kf10 = KFold(n_splits=10, shuffle=True, random_state=1)

RMSEdf = pd.DataFrame()

for i in np.arange(1, 20):
    pls = PLS(n_components=i)
    MSE = -1*cross_val_score(pls, scale(X_train), y_train, cv=kf10, scoring='neg_mean_squared_error').mean()
    RMSEdf = RMSEdf.append([pow(MSE, 0.5)])
    
RMSEdf.columns = ['MSE']
RMSEdf.reset_index(drop=True, inplace=True)
RMSEdf.index = RMSEdf.index + 1
RMSEdf

# COMMAND ----------

# Plot of PCR results
plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(RMSEdf, '-', marker = 'o', markerfacecolor='blue', markersize=8, color='green')
plt.xlabel('number of principal components', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.title('partial least squares results', fontsize=30)
plt.xlim(xmin=1)
plt.xlim(xmax=19);

# COMMAND ----------

# MAGIC %md
# MAGIC **The lowest RMSE is when I regress using 2 principal components using partial least squares.**

# COMMAND ----------

pls2 = PLS(n_components=2, scale=True)
pls2.fit(scale(X_train), y_train)

pls2pred = pls2.predict(scale(X_test))
print(np.sqrt(mean_squared_error(y_test, pls2pred)))

# COMMAND ----------

explained_variance_ratio_test = np.var(scale(X_test), axis=0) / np.sum(np.var(scale(X_test), axis=0))
EVR2 = pd.DataFrame(np.cumsum(np.round(explained_variance_ratio_test, decimals=4)*100), columns=['Explained Variance Ratio'])
EVR2.index = EVR2.index + 1
EVR2

# COMMAND ----------

