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
from sklearn.linear_model import LinearRegression
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
# MAGIC ### Principal Components Regression

# COMMAND ----------

from sklearn.preprocessing import StandardScaler, scale
import warnings
warnings.filterwarnings('ignore')
X = Hitters_1.drop(columns = ['Salary', 'Names'])
y = Hitters_1.Salary
pca = PCA()
X_scaled = pca.fit_transform(scale(X))
explained_variance_ratio = np.var(X_scaled, axis=0) / np.sum(np.var(X_scaled, axis=0))
EVR = pd.DataFrame(np.cumsum(np.round(explained_variance_ratio, decimals=4)*100), columns=['explained variance ratio'])
EVR.index = EVR.index + 1
EVR

# COMMAND ----------

# Plot of explained variance ratio
plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(EVR, '-', marker = 'o', markerfacecolor='blue', markersize=8, color='green')
plt.xlabel('number of components', fontsize=20)
plt.ylabel('explained variance ratio', fontsize=20)
plt.title('explained variance ratio', fontsize=30)
plt.xlim(xmin=-1);

# COMMAND ----------

# MAGIC %md
# MAGIC **Explained variance ratio is the percentage of variance explained in the predictors and in the response using different
# MAGIC number of components.**

# COMMAND ----------

# cross validation
from sklearn.model_selection import cross_val_score, KFold
n = len(X_scaled)
kf10 = KFold(n_splits=10, shuffle=True, random_state=1)

lm = LinearRegression()
RMSEPD = []

# Calculate RMSE with only the intercept (i.e. no principal components)
MSE = -1*cross_val_score(lm, np.ones((n,1)), y.ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()
RMSEPD.append(pow(MSE, 0.5))

# Calculate MSE using CV for the 19 principle components
for i in np.arange(1, 20):
    MSE = -1*cross_val_score(lm, X_scaled[:,:i], y.ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()
    RMSEPD.append(pow(MSE, 0.5))
RMSEdf = pd.DataFrame(data=RMSEPD, columns=['RMSE'])
RMSEdf

# COMMAND ----------

# Plot of PCR results
plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(RMSEdf, '-', marker = 'o', markerfacecolor='blue', markersize=8, color='green')
plt.xlabel('number of principal components', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.title('principal components regression results', fontsize=30)
plt.xlim(xmin=-1);

# COMMAND ----------

# MAGIC %md
# MAGIC **We see that the lowest MSE occurs for 18 principal components. This is not too different from the total number of 
# MAGIC variables(=19). So, there is not much dimension reduction to do and therefore PCR is not too useful. However, the model's RMSE drops significantly after adding just one variable and remains roughly the same which suggests that just
# MAGIC a small number of components might suffice.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split dataset into training and test dataset (and standardise them)

# COMMAND ----------

from sklearn.model_selection import train_test_split
X = Hitters_1.drop(columns = ['Salary', 'Names'])
y = Hitters_1.Salary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Principal components regression - cross validation

# COMMAND ----------

pca2 = PCA()
X_train_scaled = pca2.fit_transform(scale(X_train))
n = len(X_train_scaled)
n

# COMMAND ----------

kf10 = KFold(n_splits=10, shuffle=True, random_state=1)

lm = LinearRegression()
RMSEPD = []

# Calculate RMSE with only the intercept (i.e. no principal components)
MSE = -1*cross_val_score(lm, np.ones((n,1)), y_train.ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()
RMSEPD.append(pow(MSE, 0.5))

# Calculate MSE using CV for the 19 principle components
for i in np.arange(1, 20):
    MSE = -1*cross_val_score(lm, X_train_scaled[:,:i], y_train.ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()
    RMSEPD.append(pow(MSE, 0.5))
RMSEdf = pd.DataFrame(data=RMSEPD, columns=['RMSE'])
RMSEdf

# COMMAND ----------

# Plot of PCR results
plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(RMSEdf, '-', marker = 'o', markerfacecolor='blue', markersize=8, color='green')
plt.xlabel('number of principal components', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.title('principal components regression results - cross validation', fontsize=30)
plt.xlim(xmin=-1);

# COMMAND ----------

# MAGIC %md
# MAGIC **We notice that the smallest RMSE occurs at 5 principal components. Therefore, we will perform principal component
# MAGIC regression with 5 principal components.**

# COMMAND ----------

X_test_scaled = pca2.transform(scale(X_test))[:,:6]
lm2fit = LinearRegression().fit(X_train_scaled[:,:6], y_train)

lm2pred = lm2fit.predict(X_test_scaled)
print(np.sqrt(mean_squared_error(y_test, lm2pred)))

# COMMAND ----------

# MAGIC %md
# MAGIC **This MSE from principal components regression (PCR)  is comparable to that of ridge regression (=152308.5473577816) and 
# MAGIC lasso regression (=150198.92762434622). However, because PCR does not produce coefficient estimates like other methods,
# MAGIC it is much more difficult to interpret.**

# COMMAND ----------

explained_variance_ratio_test = np.var(X_test_scaled, axis=0) / np.sum(np.var(X_test_scaled, axis=0))
EVR6 = pd.DataFrame(np.cumsum(np.round(explained_variance_ratio_test, decimals=4)*100), columns=['Explained Variance Ratio'])
EVR6.index = EVR6.index + 1
EVR6

# COMMAND ----------

