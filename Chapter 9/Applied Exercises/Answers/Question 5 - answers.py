# Databricks notebook source
# MAGIC %md
# MAGIC We have seen that we can ft an SVM with a non-linear kernel in order
# MAGIC to perform classifcation using a non-linear decision boundary. We will
# MAGIC now see that we can also obtain a non-linear decision boundary by
# MAGIC performing logistic regression using non-linear transformations of the
# MAGIC features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Generate a data set with n = 500 and p = 2, such that the observations belong to two classes with a quadratic decision boundary
# MAGIC between them. For instance, you can do this as follows:**

# COMMAND ----------

x1 = np.random.uniform(size=500) - 0.5
x2 = np.random.uniform(size=500) - 0.5
y = 1 * (x1**2 - x2**2 > 0)

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Plot the observations, colored according to their class labels.
# MAGIC Your plot should display X1 on the x-axis, and X2 on the y-axis.**

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x=x1[y==0], y=x2[y==0], cmap='viridis', c='orange', s=500, marker='o', alpha=0.75)
plt.scatter(x=x1[y==1], y=x2[y==1], cmap='viridis', c='green', s=500, marker='o', alpha=0.75)
plt.title('observations', color='m', fontsize=30)
plt.xlabel('X1', color='orange', fontsize=20)
plt.ylabel('X2', color='green', fontsize=20)

# COMMAND ----------

# MAGIC %md
# MAGIC Clearly, there is a non-linear decision boundary.

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Fit a logistic regression model to the data, using X1 and X2 as
# MAGIC predictors.**

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# COMMAND ----------

X = pd.concat([pd.DataFrame([x1]), pd.DataFrame([x2])], axis=0).T
X.columns = ['X1', 'X2']
X.head()

# COMMAND ----------

logfit = LogisticRegression(solver='liblinear').fit(X, y.ravel())

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Apply this model to the training data in order to obtain a predicted class label for each training observation. Plot the observations, colored according to the predicted class labels. The
# MAGIC decision boundary should be linear.**

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

Y = pd.DataFrame([y]).T
df = pd.concat([Y, X], axis=1)
df.columns = ['Y', 'X1', 'X2']
df.head()

# COMMAND ----------

logpred = pd.DataFrame([logfit.predict(X)]).T
logpred.columns = ['Y_PRED']
logpred.head()

# COMMAND ----------

df = pd.concat([logpred, df], axis=1,  sort=False)
df.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x=df.X1[df.Y_PRED==0], y=df.X2[df.Y_PRED==0], cmap='viridis', c='orange', s=500, marker='o', alpha=0.75)
plt.scatter(x=df.X1[df.Y_PRED==1], y=df.X2[df.Y_PRED==1], cmap='viridis', c='green', s=500, marker='o', alpha=0.75)
plt.title('observations', color='m', fontsize=30)
plt.xlabel('X1', color='orange', fontsize=20)
plt.ylabel('X2', color='green', fontsize=20)

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(df.Y, df.Y_PRED).T, index = [0, 1], columns = [0, 1])
conf_mat

# COMMAND ----------

class_rep = classification_report(df.Y, df.Y_PRED)
print(class_rep)

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, there is a linear decision boundary.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now fit a logistic regression model to the data using non-linear
# MAGIC functions of X1 and X2 as predictors (e.g. X_1^2, X1×X2, log(X2),
# MAGIC and so forth).**

# COMMAND ----------

# MAGIC %md
# MAGIC $X_1$ x $X_2$

# COMMAND ----------

x1x2 = x1*x2

X = pd.concat([pd.DataFrame([x1]), pd.DataFrame([x2]), pd.DataFrame([x1x2])], axis=0).T
X.columns = ['X1', 'X2', 'X1 x X2']
X.head()

# COMMAND ----------

logfitX1X2 = LogisticRegression(solver='liblinear').fit(X, y.ravel())

# COMMAND ----------

Y = pd.DataFrame([y]).T
df = pd.concat([Y, X], axis=1)
df.columns = ['Y', 'X1', 'X2', 'X1 x X2']
df.head()

# COMMAND ----------

logpred = pd.DataFrame([logfitX1X2.predict(X)]).T
logpred.columns = ['Y_PRED']
logpred.head()

# COMMAND ----------

df = pd.concat([logpred, df], axis=1,  sort=False)
df.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x=df.X1[df.Y_PRED==0], y=df['X1 x X2'][df.Y_PRED==0], cmap='viridis', c='orange', s=500, marker='o', alpha=0.75)
plt.scatter(x=df.X1[df.Y_PRED==1], y=df['X1 x X2'][df.Y_PRED==1], cmap='viridis', c='green', s=500, marker='o', alpha=0.75)
plt.title('observations', color='m', fontsize=30)
plt.xlabel('X1', color='orange', fontsize=20)
plt.ylabel('X1 x X2', color='green', fontsize=20)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x=df.X2[df.Y_PRED==0], y=df['X1 x X2'][df.Y_PRED==0], cmap='viridis', c='orange', s=500, marker='o', alpha=0.75)
plt.scatter(x=df.X2[df.Y_PRED==1], y=df['X1 x X2'][df.Y_PRED==1], cmap='viridis', c='green', s=500, marker='o', alpha=0.75)
plt.title('observations', color='m', fontsize=30)
plt.xlabel('X1', color='orange', fontsize=20)
plt.ylabel('X1 x X2', color='green', fontsize=20)

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(df.Y, df.Y_PRED).T, index = [0, 1], columns = [0, 1])
conf_mat

# COMMAND ----------

class_rep = classification_report(df.Y, df.Y_PRED)
print(class_rep)

# COMMAND ----------

# MAGIC %md
# MAGIC X_1^2

# COMMAND ----------

x12 = x1**2

X = pd.concat([pd.DataFrame([x1]), pd.DataFrame([x2]), pd.DataFrame([x12])], axis=0).T
X.columns = ['X1', 'X2', 'X1^2']
X.head()

# COMMAND ----------

logfitX12 = LogisticRegression(solver='liblinear').fit(X, y.ravel())

# COMMAND ----------

Y = pd.DataFrame([y]).T
df = pd.concat([Y, X], axis=1)
df.columns = ['Y', 'X1', 'X2', 'X1^2']
df.head()

# COMMAND ----------

logpred = pd.DataFrame([logfitX12.predict(X)]).T
logpred.columns = ['Y_PRED']
logpred.head()

# COMMAND ----------

df = pd.concat([logpred, df], axis=1,  sort=False)
df.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x=df.X1[df.Y_PRED==0], y=df['X1^2'][df.Y_PRED==0], cmap='viridis', c='orange', s=500, marker='o', alpha=0.75)
plt.scatter(x=df.X1[df.Y_PRED==1], y=df['X1^2'][df.Y_PRED==1], cmap='viridis', c='green', s=500, marker='o', alpha=0.75)
plt.title('observations', color='m', fontsize=30)
plt.xlabel('X1', color='orange', fontsize=20)
plt.ylabel('X1^2', color='green', fontsize=20)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x=df.X2[df.Y_PRED==0], y=df['X1^2'][df.Y_PRED==0], cmap='viridis', c='orange', s=500, marker='o', alpha=0.75)
plt.scatter(x=df.X2[df.Y_PRED==1], y=df['X1^2'][df.Y_PRED==1], cmap='viridis', c='green', s=500, marker='o', alpha=0.75)
plt.title('observations', color='m', fontsize=30)
plt.xlabel('X1', color='orange', fontsize=20)
plt.ylabel('X1^2', color='green', fontsize=20)

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(df.Y, df.Y_PRED).T, index = [0, 1], columns = [0, 1])
conf_mat

# COMMAND ----------

class_rep = classification_report(df.Y, df.Y_PRED)
print(class_rep)

# COMMAND ----------

# MAGIC %md
# MAGIC X_2^2

# COMMAND ----------

x22 = x2**2

X = pd.concat([pd.DataFrame([x1]), pd.DataFrame([x2]), pd.DataFrame([x22])], axis=0).T
X.columns = ['X1', 'X2', 'X2^2']
X.head()

# COMMAND ----------

logfitX22 = LogisticRegression(solver='liblinear').fit(X, y.ravel())

# COMMAND ----------

Y = pd.DataFrame([y]).T
df = pd.concat([Y, X], axis=1)
df.columns = ['Y', 'X1', 'X2', 'X2^2']
df.head()

# COMMAND ----------

logpred = pd.DataFrame([logfitX12.predict(X)]).T
logpred.columns = ['Y_PRED']
logpred.head()

# COMMAND ----------

df = pd.concat([logpred, df], axis=1,  sort=False)
df.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x=df.X1[df.Y_PRED==0], y=df['X2^2'][df.Y_PRED==0], cmap='viridis', c='orange', s=500, marker='o', alpha=0.75)
plt.scatter(x=df.X1[df.Y_PRED==1], y=df['X2^2'][df.Y_PRED==1], cmap='viridis', c='green', s=500, marker='o', alpha=0.75)
plt.title('observations', color='m', fontsize=30)
plt.xlabel('X1', color='orange', fontsize=20)
plt.ylabel('X2^2', color='green', fontsize=20)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(x=df.X2[df.Y_PRED==0], y=df['X2^2'][df.Y_PRED==0], cmap='viridis', c='orange', s=500, marker='o', alpha=0.75)
plt.scatter(x=df.X2[df.Y_PRED==1], y=df['X2^2'][df.Y_PRED==1], cmap='viridis', c='green', s=500, marker='o', alpha=0.75)
plt.title('observations', color='m', fontsize=30)
plt.xlabel('X1', color='orange', fontsize=20)
plt.ylabel('X2^2', color='green', fontsize=20)

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(df.Y, df.Y_PRED).T, index = [0, 1], columns = [0, 1])
conf_mat

# COMMAND ----------

class_rep = classification_report(df.Y, df.Y_PRED)
print(class_rep)

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the non-linear boundaries don't really explain the true decision boundary well ($X1^2$ is still able to approximate the true decision boundary somewhat).

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Fit a support vector classifer to the data with X1 and X2 as
# MAGIC predictors. Obtain a class prediction for each training observation. Plot the observations, colored according to the predicted
# MAGIC class labels.**

# COMMAND ----------

from sklearn.svm import SVC

# COMMAND ----------

X = pd.concat([pd.DataFrame([x1]), pd.DataFrame([x2])], axis=0).T
X.columns = ['X1', 'X2']
X.head()

# COMMAND ----------

Y = pd.DataFrame([y]).T
df = pd.concat([Y, X], axis=1)
df.columns = ['Y', 'X1', 'X2']
df.head()

# COMMAND ----------

svmfit = SVC(kernel='linear', C=10).fit(X, y)

# COMMAND ----------

def svmplot(svc, X, y, height=0.02, buffer=0.25):
    x_min, x_max = X.X1.min()-buffer, X.X1.max()+buffer
    y_min, y_max = X.X2.min()-buffer, X.X2.max()+buffer
    xx, yy = np.meshgrid(np.arange(x_min, x_max, height), np.arange(y_min, y_max, height))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
    
plt.xkcd()
plt.figure(figsize=(25, 10))
svmplot(svmfit, X, y)
colors = ['orange' if yy == 1 else 'green' for yy in y]
plt.scatter(X.X1[:],X.X2[:], marker='o', s=250, c=colors, alpha=0.75)
plt.title('support vector classifier', color='m', fontsize=30)
plt.xlabel('X[:, 0]', color='green', fontsize=20)
plt.ylabel('X[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

svmpred = pd.DataFrame([svmfit.predict(X)]).T
svmpred.columns = ['Y_PRED']
svmpred.head()

# COMMAND ----------

df = pd.concat([svmpred, df], axis=1,)
df.head()

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(df.Y, df.Y_PRED).T, index = [0, 1], columns = [0, 1])
conf_mat

# COMMAND ----------

class_rep = classification_report(df.Y, df.Y_PRED)
print(class_rep)

# COMMAND ----------

# MAGIC %md
# MAGIC Support vector classifier with linear decision boundary doesn't provide significant improvements over logistic regression.

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Fit a SVM using a non-linear kernel to the data. Obtain a class
# MAGIC prediction for each training observation. Plot the observations,
# MAGIC colored according to the predicted class labels.**

# COMMAND ----------

svmfit = SVC(C=10, kernel='rbf', gamma=1).fit(X, y)

# COMMAND ----------

X = pd.concat([pd.DataFrame([x1]), pd.DataFrame([x2])], axis=0).T
X.columns = ['X1', 'X2']
X.head()

# COMMAND ----------

Y = pd.DataFrame([y]).T
df = pd.concat([Y, X], axis=1)
df.columns = ['Y', 'X1', 'X2']
df.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
svmplot(svmfit, X, y)
colors = ['orange' if yy == 1 else 'green' for yy in y]
plt.scatter(X.X1[:],X.X2[:], marker='o', s=250, c=colors, alpha=0.75)
plt.title('support vector machine', color='m', fontsize=30)
plt.xlabel('X[:, 0]', color='green', fontsize=20)
plt.ylabel('X[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

svmpred = pd.DataFrame([svmfit.predict(X)]).T
svmpred.columns = ['Y_PRED']
svmpred.head()

# COMMAND ----------

df = pd.concat([svmpred, df], axis=1,)
df.head()

# COMMAND ----------

conf_mat = pd.DataFrame(confusion_matrix(df.Y, df.Y_PRED).T, index = [0, 1], columns = [0, 1])
conf_mat

# COMMAND ----------

class_rep = classification_report(df.Y, df.Y_PRED)
print(class_rep)

# COMMAND ----------

# MAGIC %md
# MAGIC SVM provides an extremely significant improvement in prediction over logistic regression as well as support vector classifier.

# COMMAND ----------

# MAGIC %md
# MAGIC **i. Comment on results.**

# COMMAND ----------

# MAGIC %md
# MAGIC This question shows the power of support vector machines over other linear measures like logistic regression and support vector classifiers. This can be seen through predictive precision of different models above.