# Databricks notebook source
# MAGIC %md
# MAGIC In this problem, you will develop a model to predict whether a given car gets high or low gas mileage based on the `Auto` data set.

# COMMAND ----------

# import statistical packages
import numpy as np
import pandas as pd

# COMMAND ----------

# import data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["name"]
num_cols = list(set(Auto.columns) - set(str_cols))
Auto[str_cols] = Auto[str_cols].astype(str)
Auto[num_cols] = Auto[num_cols].astype(float)

# COMMAND ----------

Auto.head()

# COMMAND ----------

Auto.info()

# COMMAND ----------

Auto.describe()

# COMMAND ----------

Auto = Auto.drop(Auto.index[[32, 126, 330, 336, 354]])
Auto['hp'] = Auto['horsepower'].astype(float) # horsepower imports in my dataframe as string. So I convert it to float
Auto.drop(columns = 'horsepower', inplace = True)
Auto.reset_index(drop=True, inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a binary variable, `mpg01`, that contains a 1 if `mpg` contains a value above its median, and a 0 if `mpg` contains a value below its median. You can compute the median using the `median()` method of the data frame. Note you may fnd it helpful to add a column `mpg01` to the data frame by assignment. Assuming you have stored the data frame as `Auto`, this can be done as follows:**
# MAGIC <br>
# MAGIC Auto['mpg01'] = mpg01

# COMMAND ----------

mpg_median = Auto['mpg'].median()
mpg_median

# COMMAND ----------

mpg01 = pd.DataFrame(columns=['mpg01'])
Auto = pd.concat([mpg01, Auto], axis = 1)

# COMMAND ----------

Auto.head()

# COMMAND ----------

index = Auto.index

# COMMAND ----------

for i in index:
    if Auto.iloc[i]['mpg'] > mpg_median:
        Auto.at[i, 'mpg01'] = int(1)
    else:
        Auto.at[i, 'mpg01'] = int(0)

# COMMAND ----------

Auto.mpg01.dtype

# COMMAND ----------

Auto = pd.get_dummies(Auto, columns=['mpg01'], drop_first=True)

# COMMAND ----------

Auto.head()

# COMMAND ----------

type(Auto.mpg01_1)

# COMMAND ----------

cols = Auto[['mpg', 'cylinders', 'displacement', 'weight', 'acceleration',
       'year', 'origin', 'hp']]

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Explore the data graphically in order to investigate the association between `mpg01` and the other features. Which of the other features seem most likely to be useful in predicting `mpg01`? Scatterplots and boxplots may be useful tools to answer this question. Describe your fndings.**

# COMMAND ----------

for i in cols:
    plt.xkcd()
    plt.figure(figsize = (25, 10))
    sns.scatterplot(y = Auto['mpg01_1'], x = Auto[i])

# COMMAND ----------

for i in cols:
    plt.xkcd()
    plt.figure(figsize = (25, 10))
    sns.boxplot(data = [Auto['mpg01_1'], Auto[i]])
    plt.ylabel(i)
    plt.xlabel('mpg01_1')

# COMMAND ----------

import warnings
warnings.simplefilter('ignore')
plt.xkcd()
plt.figure(figsize = (25, 10))
sns.pairplot(Auto, hue = 'mpg01_1')

# COMMAND ----------

# MAGIC %md
# MAGIC There is a clear negative correlation between cylinders, weight, displacement and horsepower. There is a positive
# MAGIC correlation with acceleration.

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Split the data into a training set and a test set.**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

Auto.drop(columns=['name', 'year', 'origin'], inplace=True) # deleting name since it does not contribute towards any modelling

# COMMAND ----------

Auto.head()

# COMMAND ----------

X = Auto.drop(columns=['mpg01_1'])
y = Auto['mpg01_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Perform LDA on the training data in order to predict `mpg01`
# MAGIC using the variables that seemed most associated with `mpg01` in
# MAGIC (b). What is the test error of the model obtained?**

# COMMAND ----------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# COMMAND ----------

y_train.dtype

# COMMAND ----------

ldafit = LinearDiscriminantAnalysis().fit(X_train, y_train)

# COMMAND ----------

ldapred = ldafit.predict(X_test)

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

print(confusion_matrix(y_test, ldapred))

# COMMAND ----------

print(classification_report(y_test, ldapred))

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Perform QDA on the training data in order to predict `mpg01`
# MAGIC using the variables that seemed most associated with `mpg01` in
# MAGIC (b). What is the test error of the model obtained?**

# COMMAND ----------

qdafit = QuadraticDiscriminantAnalysis().fit(X_train, y_train)

# COMMAND ----------

qdapred = qdafit.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, qdapred))

# COMMAND ----------

print(classification_report(y_test, qdapred))

# COMMAND ----------

# MAGIC %md
# MAGIC QDA provides marginal improvement over LDA.

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Perform logistic regression on the training data in order to predict `mpg01` using the variables that seemed most associated with
# MAGIC `mpg01` in (b). What is the test error of the model obtained?**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# COMMAND ----------

glmpred = glmfit.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, glmpred))

# COMMAND ----------

print(classification_report(y_test, glmpred))

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic Regression performs the same as QDA

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Perform KNN on the training data, with several values of K, in
# MAGIC order to predict `mpg01`. Use only the variables that seemed most
# MAGIC associated with `mpg01` in (b). What test errors do you obtain?
# MAGIC Which value of K seems to perform the best on this data set?**

# COMMAND ----------

# MAGIC %md
# MAGIC Since different features are measured differently, we will need to standardise them before modelling. However, since the dataset has already been manipulated quite a bit, I will need to re-import the data and redo some of the manipulations.

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

str_cols = ["name"]
num_cols = list(set(Auto.columns) - set(str_cols))
Auto[str_cols] = Auto[str_cols].astype(str)
Auto[num_cols] = Auto[num_cols].astype(float)

# COMMAND ----------

Auto = Auto.drop(Auto.index[[32, 126, 330, 336, 354]])
Auto['hp'] = Auto['horsepower'].astype(float) # horsepower imports in my dataframe as string. So I convert it to float
Auto.drop(columns = 'horsepower', inplace = True)
Auto.reset_index(drop=True, inplace = True)

# COMMAND ----------

mpg_median = Auto['mpg'].median()
mpg_median

# COMMAND ----------

mpg01 = pd.DataFrame(columns=['mpg01'])
Auto = pd.concat([mpg01, Auto], axis = 1)

# COMMAND ----------

index = Auto.index

# COMMAND ----------

for i in index:
    if Auto.iloc[i]['mpg'] > mpg_median:
        Auto.at[i, 'mpg01'] = int(1)
    else:
        Auto.at[i, 'mpg01'] = int(0)

# COMMAND ----------

Auto = pd.get_dummies(Auto, columns=['mpg01'], drop_first=True)

# COMMAND ----------

Auto.drop(columns=['name', 'year', 'origin'], inplace=True) # deleting name since it does not contribute towards any modelling

# COMMAND ----------

Auto.head()

# COMMAND ----------

Auto.shape

# COMMAND ----------

pf = pd.DataFrame()
for i in Auto.columns[:-1]:
    pf = pf.append([Auto[i].var()])


plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(pf.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC Whoa! We need to standardise the variables!

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scaler = StandardScaler()

# COMMAND ----------

import warnings
warnings.simplefilter('ignore')
scaler.fit(Auto.drop(columns='mpg01_1', axis=1))

# COMMAND ----------

scaled_features = scaler.transform(Auto.drop(columns='mpg01_1',axis=1))

# COMMAND ----------

Auto_scaled = pd.DataFrame(scaled_features,columns=Auto.columns[:-1])
Auto_scaled.head()

# COMMAND ----------

pf = pd.DataFrame()
for i in Auto_scaled.columns[:-1]:
    pf = pf.append([Auto_scaled[i].var()])


plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(pf.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC Looks great! Now, I can proceed to the modelling phase.

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

# COMMAND ----------

knnpred = knn.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knnpred))

# COMMAND ----------

print(classification_report(y_test, knnpred))

# COMMAND ----------

error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25,10))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, there is no point in modelling for K > 3, since the error rate increases drastically. This is primarily 
# MAGIC because the Bayesian Boundary is likely to be non-linear. Just to check, I will model with K = 3. However, I am not expecting any significant improvement in accuracy over K = 1.

# COMMAND ----------

knn3 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# COMMAND ----------

knnpred3 = knn3.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knnpred3))

# COMMAND ----------

print(classification_report(y_test, knnpred3))

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, there is no significant change in model prediction accuracy. Now, for fun, let me do K = 11, which will highlight issues with overfitting a non-linear Bayesian Boundary.

# COMMAND ----------

knn11 = KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train)

# COMMAND ----------

knnpred11 = knn11.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knnpred11))

# COMMAND ----------

print(classification_report(y_test, knnpred11))

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, there is a drastic reduction in model prediction accuracy due to overfitting of test data.

# COMMAND ----------

