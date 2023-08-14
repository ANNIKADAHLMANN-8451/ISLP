# Databricks notebook source
# MAGIC %md
# MAGIC This question should be answered using the Weekly data set, which is part of the ISLP package. This data is similar in nature to the Smarket data from this chapter’s lab, except that it contains 1,089 weekly returns for 21 years, from the beginning of 1990 to the end of 2010.

# COMMAND ----------

# general imports
import numpy as np
import pandas as pd

# COMMAND ----------

# import data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Weekly.csv"
Weekly = spark.read.option("header", "true").csv(url).toPandas()
Weekly.set_index('_c0', inplace=True)

float_cols = ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today"]
int_cols = ['Year']
str_cols = ["Direction"]
Weekly[float_cols] = Weekly[float_cols].astype(float)
Weekly[int_cols] = Weekly[int_cols].astype(int)
Weekly[str_cols] = Weekly[str_cols].astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Produce some numerical and graphical summaries of the `Weekly` data. Do there appear to be any patterns?**

# COMMAND ----------

Weekly.head()

# COMMAND ----------

Weekly.info()

# COMMAND ----------

Weekly.describe()

# COMMAND ----------

Weekly.cov()

# COMMAND ----------

Weekly.corr()

# COMMAND ----------

sns.pairplot(Weekly)

# COMMAND ----------

# MAGIC %md
# MAGIC There appears to be a strong discernable relationship between Year and Volume. This can be seen through correlation 
# MAGIC between Year and Volume (~0.84) as well as the graph where is a discernable pattern through which one can draw a regression
# MAGIC line. There does not appear to be a discernable relation between other variables.

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Use the full data set to perform a logistic regression with `Direction` as the response and the fve lag variables plus `Volume` as predictors. Use the summary function to print the results. Do any of the predictors appear to be statistically signifcant? If so, which ones?**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

X = Weekly.drop(columns='Direction', axis=1)
y = Weekly['Direction']

# COMMAND ----------

glmfit = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X, y)

# COMMAND ----------

coefficients = glmfit.coef_

# COMMAND ----------

print(coefficients)

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Compute the confusion matrix and overall fraction of correct predictions. Explain what the confusion matrix is telling you about the types of mistakes made by logistic regression.**

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

glmpred = glmfit.predict(X)

# COMMAND ----------

print(confusion_matrix(y, glmpred))

# COMMAND ----------

print(classification_report(y, glmpred))

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic regression predicts the values REALLY well given it just turns out 3 Type I errors (false positives). This is not surprising given the test data is the same as training data.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Now ft the logistic regression model using a training data period from 1990 to 2008, with `Lag2` as the only predictor. Compute the confusion matrix and the overall fraction of correct predictions for the held out data (that is, the data from 2009 and 2010).**

# COMMAND ----------

X_train = Weekly[Weekly['Year'] < 2009].drop(columns=['Direction','Lag1', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today', 'Year'], axis=1)

# COMMAND ----------

X_test = Weekly[Weekly['Year'] >= 2009].drop(columns=['Direction','Lag1', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today', 'Year'], axis=1)

# COMMAND ----------

y_train = np.ravel(Weekly[Weekly['Year'] < 2009].drop(columns=['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today', 'Year']))

# COMMAND ----------

y_test = np.ravel(Weekly[Weekly['Year'] >= 2009].drop(columns=['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today', 'Year']))

# COMMAND ----------

glmfit1 = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# COMMAND ----------

glmfit1pred = glmfit1.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, glmfit1pred))

# COMMAND ----------

print(confusion_matrix(y_test, glmfit1pred))

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like I have hit the jackpot here! Just kidding. This model is more realistic given the increase in Type I and Type II errors.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Repeat (d) using LDA.**

# COMMAND ----------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# COMMAND ----------

lda = LinearDiscriminantAnalysis().fit(X_train, y_train)

# COMMAND ----------

ldapred = lda.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, ldapred))

# COMMAND ----------

print(classification_report(y_test, ldapred))

# COMMAND ----------

# MAGIC %md
# MAGIC LDA does not provide any significant improvement over Logistic Regression. In fact, its accuracy is the same as that of
# MAGIC Logistic Regression.

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Repeat (d) using QDA.**

# COMMAND ----------

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# COMMAND ----------

qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)

# COMMAND ----------

qdapred = qda.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, qdapred))

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore') # I use this to ignore a warning that states the classification report is unable to calculate F-Scores, which is not required for this instance
print(classification_report(y_test, qdapred))

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see QDA actually improves upon true positives and false positives, but it comes at a heavy cost of being able
# MAGIC to predict true negatives and false negatives. Whether this is critical will depend on context. For examples, banks assessing
# MAGIC the ability of the model to predict potential delinquencis might be alright with it since they are likely to prioritise
# MAGIC false positives (people who will default, but the model declares them otherwise) over false negatives (people will not
# MAGIC default, but the model declares them otherwise).

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Repeat (d) using KNN with K = 1.**

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

# MAGIC %md
# MAGIC The precision of K-nearest neighbours reduces. This is because at K = 1, the classifier is highly non-linear and 
# MAGIC accuracy results from Logistic Regression and LDA suggest that the classifier is likely to be linear. As such, the 
# MAGIC K-nearest neighbours likely overfits the test data.

# COMMAND ----------

# MAGIC %md
# MAGIC **i. Which of these methods appears to provide the best results on this data?**

# COMMAND ----------

# MAGIC %md
# MAGIC The accuracy reports suggest that Logistic Regression and Linear Discriminant Analysis provide the best results
# MAGIC on this data.

# COMMAND ----------

