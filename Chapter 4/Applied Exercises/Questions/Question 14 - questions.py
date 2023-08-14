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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Explore the data graphically in order to investigate the association between `mpg01` and the other features. Which of the other features seem most likely to be useful in predicting `mpg01`? Scatterplots and boxplots may be useful tools to answer this question. Describe your fndings.**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Split the data into a training set and a test set.**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Perform LDA on the training data in order to predict `mpg01`
# MAGIC using the variables that seemed most associated with `mpg01` in
# MAGIC (b). What is the test error of the model obtained?**

# COMMAND ----------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Perform QDA on the training data in order to predict `mpg01`
# MAGIC using the variables that seemed most associated with `mpg01` in
# MAGIC (b). What is the test error of the model obtained?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Perform logistic regression on the training data in order to predict `mpg01` using the variables that seemed most associated with
# MAGIC `mpg01` in (b). What is the test error of the model obtained?**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# TODO: your response here

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

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

# TODO: your response here