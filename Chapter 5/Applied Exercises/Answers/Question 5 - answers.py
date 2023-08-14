# Databricks notebook source
# MAGIC %md
# MAGIC In Chapter 4, we used logistic regression to predict the probability of
# MAGIC `default` using `income` and `balance` on the `Default` data set. We will
# MAGIC now estimate the test error of this logistic regression model using the
# MAGIC validation set approach. Do not forget to set a random seed before
# MAGIC beginning your analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

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

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Default.csv"
Default = spark.read.option("header", "true").csv(url).toPandas()
Default.set_index('_c0', inplace=True)

float_cols = ["balance", "income"]
str_cols = ["default", "student"]
Default[float_cols] = Default[float_cols].astype(float)
Default[str_cols] = Default[str_cols].astype(str)

# COMMAND ----------

Default.head()

# COMMAND ----------

Default.info()

# COMMAND ----------

dfX = Default[['student', 'balance','income']]
dfX = pd.get_dummies(data = dfX, drop_first=True)
dfy = Default['default']

# COMMAND ----------

dfX.head()

# COMMAND ----------

dfy.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Fit a logistic regression model that uses `income` and `balance` to
# MAGIC predict `default`.**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

X = dfX[['income', 'balance']]
y = dfy

# COMMAND ----------

glmfit = LogisticRegression(solver = 'liblinear').fit(X, y)

# COMMAND ----------

glmfit.coef_

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Using the validation set approach, estimate the test error of this
# MAGIC model. In order to do this, you must perform the following steps:**
# MAGIC - i. Split the sample set into a training set and a validation set.
# MAGIC - ii. Fit a multiple logistic regression model using only the training observations.
# MAGIC - iii. Obtain a prediction of `default` status for each individual in the validation set by computing the posterior probability of default for that individual, and classifying the individual to the default category if the posterior probability is greater than 0.5.
# MAGIC - iv. Compute the validation set error, which is the fraction of the observations in the validation set that are misclassifed.

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X = dfX[['income', 'balance']]
y = dfy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

print("X_train, ", X_train.shape, "y_train, ", y_train.shape, "X_test: ", X_test.shape, "y_test: ", y_test.shape)

# COMMAND ----------

glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# COMMAND ----------

glmpred = glmfit.predict(X_test)

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# COMMAND ----------

conf_mat = confusion_matrix(y_test, glmpred)
conf_mat

# COMMAND ----------

round((conf_mat[0][1] + conf_mat[1][0]) / y_train.shape[0], 4)

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Repeat the process in (b) three times, using three diferent splits
# MAGIC of the observations into a training set and a validation set. Comment on the results obtained.**

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("X_train, ", X_train.shape, "y_train, ", y_train.shape, "X_test: ", X_test.shape, "y_test: ", y_test.shape)
glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)
glmpred = glmfit.predict(X_test)
conf_mat = confusion_matrix(y_test, glmpred)
round((conf_mat[0][1] + conf_mat[1][0]) / y_train.shape[0], 4)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print("X_train, ", X_train.shape, "y_train, ", y_train.shape, "X_test: ", X_test.shape, "y_test: ", y_test.shape)
glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)
glmpred = glmfit.predict(X_test)
conf_mat = confusion_matrix(y_test, glmpred)
round((conf_mat[0][1] + conf_mat[1][0]) / y_train.shape[0], 4)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)
print("X_train, ", X_train.shape, "y_train, ", y_train.shape, "X_test: ", X_test.shape, "y_test: ", y_test.shape)
glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)
glmpred = glmfit.predict(X_test)
conf_mat = confusion_matrix(y_test, glmpred)
round((conf_mat[0][1] + conf_mat[1][0]) / y_train.shape[0], 4)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking for multiple splits

# COMMAND ----------

sample = np.linspace(start = 0.05, stop = 0.95, num = 20)

# COMMAND ----------

sample

# COMMAND ----------

X = dfX[['income', 'balance']]
y = dfy
confpd = pd.DataFrame()
for i in sample:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=42)
    print("X_train, ", X_train.shape, "y_train, ", y_train.shape, "X_test: ", X_test.shape, "y_test: ", y_test.shape)
    glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)
    glmpred = glmfit.predict(X_test)
    conf_mat = confusion_matrix(y_test, glmpred)
    sum = round((conf_mat[0][1] + conf_mat[1][0]) / y_train.shape[0], 4)
    confpd = confpd.append([sum])

# COMMAND ----------

confpd.reset_index(drop=True, inplace=True)

# COMMAND ----------

confpd.columns = ['Error']

# COMMAND ----------

confpd.head()

# COMMAND ----------

confpd.mean()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(confpd, marker = 'o', markersize = 10)
plt.title("split% vs error rates")
plt.ylabel("error rates")
plt.xlabel("split%")

# COMMAND ----------

# MAGIC %md
# MAGIC We notice that the error rate asymptotically settle around ~0.62, but the growth really begins to plateau around 0.2.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Now consider a logistic regression model that predicts the probability of `default` using `income`, `balance`, and a dummy variable
# MAGIC for `student`. Estimate the test error for this model using the validation set approach. Comment on whether or not including a
# MAGIC dummy variable for `student` leads to a reduction in the test error
# MAGIC rate.**

# COMMAND ----------

X = dfX # no need to change since dfX already incorporates the dummy variable transformation for 'student'
y = dfy

# COMMAND ----------

# MAGIC %md
# MAGIC Using the validation set approach

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# COMMAND ----------

glmpred = glmfit.predict(X_test)

# COMMAND ----------

confusion_matrix(y_test, glmpred)

# COMMAND ----------

round((conf_mat[0][1] + conf_mat[1][0]) / y_train.shape[0], 4)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking for multiple splits

# COMMAND ----------

sample = np.linspace(start = 0.05, stop = 0.95, num = 20)
sample

# COMMAND ----------

X = dfX
y = dfy
confpd = pd.DataFrame()
for i in sample:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=42)
    print("X_train, ", X_train.shape, "y_train, ", y_train.shape, "X_test: ", X_test.shape, "y_test: ", y_test.shape)
    glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)
    glmpred = glmfit.predict(X_test)
    conf_mat = confusion_matrix(y_test, glmpred)
    sum = round((conf_mat[0][1] + conf_mat[1][0]) / y_train.shape[0], 4)
    confpd = confpd.append([sum])

# COMMAND ----------

confpd.reset_index(drop=True, inplace=True)
confpd.columns = ['Error']
confpd.head()

# COMMAND ----------

confpd.mean()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(confpd, marker = 'o', markersize = 10)
plt.title("split% vs error rates")
plt.ylabel("error rates")
plt.xlabel("split%")

# COMMAND ----------

# MAGIC %md
# MAGIC We notice the same graph as that for logit without the dummy variable. So, we can conclude that the dummy variable
# MAGIC does not lead to a reduction in the test error rate