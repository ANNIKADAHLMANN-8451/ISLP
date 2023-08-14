# Databricks notebook source
# MAGIC %md
# MAGIC  Using the `Boston` data set, fit classifcation models in order to predict
# MAGIC whether a given suburb has a crime rate above or below the median.
# MAGIC Explore logistic regression, LDA, naive Bayes, and KNN models using
# MAGIC various subsets of the predictors. Describe your fndings.
# MAGIC <br>
# MAGIC *Hint: You will have to create the response variable yourself, using the
# MAGIC variables that are contained in the `Boston` data set.*

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

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas()
Boston.set_index('SlNo', inplace=True)

int_cols = ['chas', 'rad', 'tax']
float_cols = list(set(Boston.columns) - set(int_cols))
Boston[int_cols] = Boston[int_cols].astype(int)
Boston[float_cols] = Boston[float_cols].astype(float)

# COMMAND ----------

Boston.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **Calculating median crime rate**

# COMMAND ----------

crim_median = Boston['crim'].median()

# COMMAND ----------

crim_median

# COMMAND ----------

# MAGIC %md
# MAGIC **Adding classification data crim1**

# COMMAND ----------

crim1 = pd.DataFrame(columns=['crim1'])

# COMMAND ----------

Boston = pd.concat([crim1, Boston], axis = 1)

# COMMAND ----------

Boston.head()

# COMMAND ----------

index = Boston.index

# COMMAND ----------

for i in index:
    if Boston.loc[i]['crim'] > crim_median:
        Boston.at[i, 'crim1'] = 1
    else:
        Boston.at[i, 'crim1'] = 0

# COMMAND ----------

type(Boston['crim1'])

# COMMAND ----------

Boston

# COMMAND ----------

Boston.crim1.dtype

# COMMAND ----------

# MAGIC %md
# MAGIC *As we can see the data type of Boston.crim1 is not in any recognisable format which will cause problems later on. So, we will first have to convert the data type of Boston into a dummy variable.*

# COMMAND ----------

Boston = pd.get_dummies(Boston, columns=['crim1'], drop_first=True)

# COMMAND ----------

Boston.head(25)

# COMMAND ----------

Boston.crim1_1.dtype

# COMMAND ----------

# MAGIC %md
# MAGIC *We see there is a new column Boston.crim1_1 with integral digits. We will use this column for modelling.*

# COMMAND ----------

# MAGIC %md
# MAGIC **Important question: What is the true nature of distribution of each independent variable?**

# COMMAND ----------

# MAGIC %md
# MAGIC *This is important because an underlying assumption of LDA and QDA is that the marginal distribution of each variable is normal. Non-normality can significantly reduce the predictive performance of LDA and QDA.*

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
for i in Boston.columns:
    plt.xkcd()
    plt.figure(figsize = (25, 10))
    sns.distplot(Boston[i])

# COMMAND ----------

# MAGIC %md
# MAGIC *As we can see, amongst non-categorical data, only rm (and somewhat medv) have a normal distribution. The question then is - how strongly are these and other non-normally distributed predictors correlated with crime rates? We can use the correlation to determine if we could potentially keep these predictors. First, let me perform predictions using ALL predictors.*

# COMMAND ----------

# MAGIC %md
# MAGIC **Dividing the dataframe into training and test data**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X = Boston.drop(columns='crim1_1')
y = Boston['crim1_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Logistic Regression**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

glmfit = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# COMMAND ----------

glmpred = glmfit.predict(X_test)

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

print(confusion_matrix(y_test, glmpred))

# COMMAND ----------

print(classification_report(y_test, glmpred))

# COMMAND ----------

# MAGIC %md
# MAGIC *93% overall (unweighted) precision is great! This means we correctly predicted the crime rates in 93% of cases. Let's explore where the model is inaccurate. In ~7.23% of the cases, it wrongly classifies neighbourhoods with their crime rates. Delving deeper using the classification report, we see that the issue stems from (relatively) low precision in those neighbourhoods where the crim was lower than median. This might be an issue from a practical standpoint since classifying some less-crime affected areas would mean the government would deploy disproportionately large police force at the expense of other areas with higher than median crime rates.*

# COMMAND ----------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

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
# MAGIC *As we can see, using LDA reduces the precision of the model.*

# COMMAND ----------

qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)

# COMMAND ----------

qdapred = qda.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, qdapred))

# COMMAND ----------

print(classification_report(y_test, qdapred))

# COMMAND ----------

# MAGIC %md
# MAGIC *QDA improves upon the results of logistic regression and could be considered as the contender for Boston's top model.*

# COMMAND ----------

# MAGIC %md
# MAGIC **K-Nearest Neighbours**

# COMMAND ----------

# MAGIC %md
# MAGIC *We will need to standardise the predictors since they all measure differently.*

# COMMAND ----------

# we will need to import data again
url = "/Users/arpanganguli/Documents/Professional/Finance/ISLR/Datasets/Boston.csv"
Boston = pd.read_csv(url)
crim_median = Boston['crim'].median()
crim1 = pd.DataFrame(columns=['crim1'])
Boston = pd.concat([crim1, Boston], axis = 1)
Boston.drop(columns='SlNo', inplace=True)
index = Boston.index
for i in index:
    if Boston.iloc[i]['crim'] > crim_median:
        Boston.at[i, 'crim1'] = 1
    else:
        Boston.at[i, 'crim1'] = 0
Boston = pd.get_dummies(Boston, columns=['crim1'], drop_first=True)

# COMMAND ----------

Boston.head(25)

# COMMAND ----------

# MAGIC %md
# MAGIC *Let me check the variances of each predictor.*

# COMMAND ----------

pf = pd.DataFrame()
for i in Boston.columns[:-1]:
    pf = pf.append([Boston[i].var()])

pf.columns = ['var']
plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(pf['var'].reset_index())

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scaler = StandardScaler()

# COMMAND ----------

scaler.fit(Boston.drop(columns='crim1_1', axis = 1).astype(float))

# COMMAND ----------

scaled_features = scaler.transform(Boston.drop(columns='crim1_1', axis = 1).astype(float))

# COMMAND ----------

Boston_scaled = pd.DataFrame(scaled_features, columns=Boston.columns[:-1])

# COMMAND ----------

Boston_scaled.head()

# COMMAND ----------

# MAGIC %md
# MAGIC *Checking the variances of each predictor in the scaled dataframe.*

# COMMAND ----------

pf = pd.DataFrame()
for i in Boston_scaled.columns[:-1]:
    pf = pf.append([Boston_scaled[i].var()])


plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(pf.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC *Okay, this looks good!*

# COMMAND ----------

# MAGIC %md
# MAGIC *Let me visually examine the error rate for different values of K*

# COMMAND ----------

X = Boston_scaled
y = Boston['crim1_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

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
# MAGIC *So we can see, the error rate is lowest for K = 2 and then keeps increasing thereafter. So, I will perform KNNs for K = 1, 2 and 39 (highest error rate)*

# COMMAND ----------

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train,y_train)
knnpred1 = knn1.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knnpred1))

# COMMAND ----------

print(classification_report(y_test, knnpred1))

# COMMAND ----------

knn2 = KNeighborsClassifier(n_neighbors=2)
knn2.fit(X_train,y_train)
knnpred2 = knn2.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knnpred2))

# COMMAND ----------

knn39 = KNeighborsClassifier(n_neighbors=39)
knn39.fit(X_train,y_train)
knnpred39 = knn39.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knnpred39))

# COMMAND ----------

print(classification_report(y_test, knnpred39))

# COMMAND ----------

# MAGIC %md
# MAGIC *As we can see, we get the best overall precision at K = 2 (92%) and the worst precision at K = 39 (84%). However, the best precision for K-Means is a shade lower than that of QDA. So, QDA is the best classifier, given we use ALL predictors.*

# COMMAND ----------

# MAGIC %md
# MAGIC **So, how about using a subset of predictors?**

# COMMAND ----------

# MAGIC %md
# MAGIC *First, we check the pairplots and the correlation matrix.*

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.pairplot(Boston)

# COMMAND ----------

round(Boston.corr()*100, 2)

# COMMAND ----------

# MAGIC %md
# MAGIC *Assuming an arbitrary correlation cutoff of 35% (in absolute value) and using some qualitative judgment (such as disregarding 'nox') the most correlated predictors are 'indus', 'age', 'dis', 'tax', 'black' and 'lstat'. These should give a healthy idea about some prime factors (and reverse factors too!) for crime.*

# COMMAND ----------

# MAGIC %md
# MAGIC *Two points of note are in order here:*<br>
# MAGIC *1. The selections of these columns are arbitrary and conditional upon my personal bias. So, the reader is expected to play around more with different subsets and explore for themselves.*<br>
# MAGIC *2. I have not considered multicollinearity amongst different predictors, something that should be done. At the moment, I have just 'eyeballed' the multicollinearity (such as those areas with large 'indus'(non-retail businesses) are likely to have large 'nox' (nitric oxides concentration (parts per 10 million) because of fumes from these businesses.*

# COMMAND ----------

Boston1 = Boston.drop(columns=['crim', 'zn', 'chas', 'nox', 'rad', 'medv'])

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
for i in Boston.columns:
    plt.xkcd()
    plt.figure(figsize = (25, 10))
    sns.distplot(Boston[i])

# COMMAND ----------

sns.pairplot(Boston1)

# COMMAND ----------

# MAGIC %md
# MAGIC **Splitting the Boston1 dataset into training and test data**

# COMMAND ----------

X = Boston1.drop(columns='crim1_1', axis=1)
y = Boston1['crim1_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Logistic Regression**

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
# MAGIC **Linear Discriminant Analysis**

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
# MAGIC **Quadratic Discriminant Analysis**

# COMMAND ----------

qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)

# COMMAND ----------

qdapred = qda.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, qdapred))

# COMMAND ----------

print(classification_report(y_test, qdapred))

# COMMAND ----------

# MAGIC %md
# MAGIC **K-Nearest Neighbours**

# COMMAND ----------

pf = pd.DataFrame()
for i in Boston1.columns[:-1]:
    pf = pf.append([Boston1[i].var()])

pf.columns = ['var']
plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(pf['var'].reset_index())

# COMMAND ----------

scaler = StandardScaler()

# COMMAND ----------

scaler.fit(Boston1.drop(columns='crim1_1', axis = 1).astype(float))

# COMMAND ----------

scaled_features = scaler.transform(Boston1.drop(columns='crim1_1', axis = 1).astype(float))

# COMMAND ----------

Boston1_scaled = pd.DataFrame(scaled_features, columns=Boston1.columns[:-1])

# COMMAND ----------

Boston1_scaled.head()

# COMMAND ----------

# MAGIC %md
# MAGIC *Checking the variances of each predictor in the Boston1 scaled dataframe.*

# COMMAND ----------

pf = pd.DataFrame()
for i in Boston1_scaled.columns[:-1]:
    pf = pf.append([Boston1_scaled[i].var()])


plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(pf.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC *Looks good!*

# COMMAND ----------

X = Boston1
y = Boston['crim1_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
# MAGIC *Let me check the KNN predictions for K = 2 (lowest error rate) and K = 21 (highest error rate)*

# COMMAND ----------

knn2 = KNeighborsClassifier(n_neighbors=2)
knn2.fit(X_train,y_train)
knnpred2 = knn2.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knnpred2))

# COMMAND ----------

print(classification_report(y_test, knnpred2))

# COMMAND ----------

knn21 = KNeighborsClassifier(n_neighbors=21)
knn21.fit(X_train,y_train)
knnpred21 = knn21.predict(X_test)

# COMMAND ----------

print(confusion_matrix(y_test, knnpred21))

# COMMAND ----------

print(classification_report(y_test, knnpred21))

# COMMAND ----------

# MAGIC %md
# MAGIC *As we can see, taking the subset, KNN(K=2) provides the best precision for all models. Likewise, we could conduct further tests and check for different subsets of the data.*