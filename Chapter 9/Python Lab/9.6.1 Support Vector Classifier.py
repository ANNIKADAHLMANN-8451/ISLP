# Databricks notebook source
# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

np.random.seed(1)

# COMMAND ----------

X = np.random.normal(size=(20,2))
X

# COMMAND ----------

y_1 = np.array([1]*10)
y_minus1 = np.array([-1]*10)
y = np.concatenate([y_1, y_minus1])
y

# COMMAND ----------

X[y==1] += 1
X

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
colors = ['orange' if yy == 1 else 'green' for yy in y]
plt.scatter(X[:,0][:],X[:,1][:], marker='o', s=1000, c=colors)
plt.title('are the two classes linearly separable?', color='m', fontsize=30)
plt.xlabel('X[:, 0]', color='green', fontsize=20)
plt.ylabel('X[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

# MAGIC %md
# MAGIC **Therefore, the two classes are not linearly separable.**

# COMMAND ----------

# MAGIC %md
# MAGIC **Support vector classifier**

# COMMAND ----------

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
svmfit10 = SVC(kernel='linear', C=10).fit(X, y)

# COMMAND ----------

def svmplot(svc, X, y, height=0.02, buffer=0.25):
    x_min, x_max = X[:, 0].min()-buffer, X[:, 0].max()+buffer
    y_min, y_max = X[:, 1].min()-buffer, X[:, 1].max()+buffer
    xx, yy = np.meshgrid(np.arange(x_min, x_max, height), np.arange(y_min, y_max, height))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
    
plt.xkcd()
plt.figure(figsize=(25, 10))
svmplot(svmfit10, X, y)
colors = ['orange' if yy == 1 else 'green' for yy in y]
plt.scatter(X[:,0][:],X[:,1][:], marker='o', s=1000, c=colors)
plt.title('support vector classifier', color='m', fontsize=30)
plt.xlabel('X[:, 0]', color='green', fontsize=20)
plt.ylabel('X[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

svmfit10.support_ # these are the support vectors

# COMMAND ----------

conf_mat10 = pd.DataFrame(confusion_matrix(y, svmfit10.predict(X)).T, index = svmfit10.classes_, columns = svmfit10.classes_)
conf_mat10

# COMMAND ----------

class_mat10 = classification_report(y, svmfit10.predict(X))
print(class_mat10)

# COMMAND ----------

svmfit001 = SVC(kernel='linear', C=0.001).fit(X, y) # using smaller value of cost vector

plt.figure(figsize=(25, 10))
svmplot(svmfit001, X, y)
colors = ['orange' if yy == 1 else 'green' for yy in y]
plt.scatter(X[:,0][:],X[:,1][:], marker='o', s=1000, c=colors)
plt.title('support vector classifier', color='m', fontsize=30)
plt.xlabel('X[:, 0]', color='green', fontsize=20)
plt.ylabel('X[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

svmfit001.support_ # these are the support vectors

# COMMAND ----------

# MAGIC %md
# MAGIC *If we use smaller cost vector, there are a larger number of support vectors used because the margin is now wider.*

# COMMAND ----------

conf_mat001 = pd.DataFrame(confusion_matrix(y, svmfit001.predict(X)).T, index = svmfit001.classes_, columns = svmfit001.classes_)
conf_mat001

# COMMAND ----------

class_mat001 = classification_report(y, svmfit001.predict(X))
print(class_mat001)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using cross-validation to find the optimal cost

# COMMAND ----------

from sklearn.model_selection import GridSearchCV as GSV

# COMMAND ----------

cost_range = [{'C': np.linspace(0.001, 20, 1000)}]
cost_cv= GSV(SVC(kernel='linear'), cost_range, cv=10, scoring='accuracy', return_train_score=True).fit(X, y)

# COMMAND ----------

best_params = cost_cv.best_params_
best_params['C']

# COMMAND ----------

# MAGIC %md
# MAGIC **GridSearchCV suggests that the best results are obtained at C=0.16115215215215214.**

# COMMAND ----------

X_test = np.random.normal(size=(20,2))
X_test

# COMMAND ----------

y_1_test = np.array([1]*10)
y_minus1_test = np.array([-1]*10)
y_test = np.concatenate([y_1_test, y_minus1_test])
y

# COMMAND ----------

X_test[y_test==1] += 1
X_test

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
colors = ['orange' if yy == 1 else 'green' for yy in y_test]
plt.scatter(X_test[:,0][:],X_test[:,1][:], marker='o', s=1000, c=colors)
plt.title('are the two classes linearly separable?', color='m', fontsize=30)
plt.xlabel('X[:, 0]', color='green', fontsize=20)
plt.ylabel('X[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

np.random.seed(101)
X_test = np.random.normal(size=(20,2))
y_test = np.random.choice([-1,1], 20)
X_test[y_test == 1] = X_test[y_test == 1]-1

plt.xkcd()
plt.figure(figsize=(25, 10))
colors = ['orange' if yy == 1 else 'green' for yy in y_test]
plt.scatter(X_test[:,0][:],X_test[:,1][:], marker='o', s=1000, c=colors)
plt.title('are the two classes linearly separable?', color='m', fontsize=30)
plt.xlabel('X_test[:, 0]', color='green', fontsize=20)
plt.ylabel('X_test[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

svmfit_test = SVC(C=best_params['C'], kernel='linear').fit(X, y)

plt.figure(figsize=(25, 10))
svmplot(svmfit_test, X_test, y_test)
colors = ['orange' if yy == 1 else 'green' for yy in y_test]
plt.scatter(X_test[:,0][:],X_test[:,1][:], marker='o', s=1000, c=colors)
plt.title('support vector classifier', color='m', fontsize=30)
plt.xlabel('X_test[:, 0]', color='green', fontsize=20)
plt.ylabel('X_test[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

conf_mat_test = pd.DataFrame(confusion_matrix(y_test, svmfit_test.predict(X_test)), index=svmfit_test.classes_, columns=svmfit_test.classes_)
conf_mat_test

# COMMAND ----------

class_mat = classification_report(y_test, svmfit_test.predict(X_test))
print(class_mat)

# COMMAND ----------

svmfit_test001 = SVC(C=10, kernel='linear')
svmfit_test001.fit(X, y)

plt.figure(figsize=(25, 10))
svmplot(svmfit_test, X_test, y_test)
colors = ['orange' if yy == 1 else 'green' for yy in y_test]
plt.scatter(X_test[:,0][:],X_test[:,1][:], marker='o', s=1000, c=colors)
plt.title('support vector classifier', color='m', fontsize=30)
plt.xlabel('X_test[:, 0]', color='green', fontsize=20)
plt.ylabel('X_test[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

conf_mat_test = pd.DataFrame(confusion_matrix(y_test, svmfit_test001.predict(X_test)), index=svmfit_test001.classes_, columns=svmfit_test001.classes_)
conf_mat_test

# COMMAND ----------

class_mat_test001 = classification_report(y_test, svmfit_test.predict(X_test))
print(class_mat_test001)

# COMMAND ----------

# MAGIC %md
# MAGIC **Therefore, we see that C=10 provides worse results that C=0.16115215215215214. Using other values of C provide the same result.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Support Vector classifier with linearly separable classes

# COMMAND ----------

X = np.random.normal(size=(20,2))
X

# COMMAND ----------

y_1 = np.array([1]*10)
y_minus1 = np.array([-1]*10)
y = np.concatenate([y_1, y_minus1])
y

# COMMAND ----------

X[y==1] += 0.5
X

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
colors = ['orange' if yy == 1 else 'green' for yy in y]
plt.scatter(X[:,0][:],X[:,1][:], marker='o', s=1000, c=colors)
plt.title('are the two classes linearly separable?', color='m', fontsize=30)
plt.xlabel('X[:, 0]', color='green', fontsize=20)
plt.ylabel('X[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

svmfit10 = SVC(kernel='linear', C=10).fit(X, y)
  
plt.xkcd()
plt.figure(figsize=(25, 10))
svmplot(svmfit10, X, y)
colors = ['orange' if yy == 1 else 'green' for yy in y]
plt.scatter(X[:,0][:],X[:,1][:], marker='o', s=1000, c=colors)
plt.title('support vector classifier', color='m', fontsize=30)
plt.xlabel('X[:, 0]', color='green', fontsize=20)
plt.ylabel('X[:, 1]', color='orange', fontsize=20)

# COMMAND ----------

svmfit10.support_ # these are the support vectors

# COMMAND ----------

conf_mat10 = pd.DataFrame(confusion_matrix(y, svmfit10.predict(X)).T, index = svmfit10.classes_, columns = svmfit10.classes_)
conf_mat10

# COMMAND ----------

class_mat10 = classification_report(y, svmfit10.predict(X))
print(class_mat10)

# COMMAND ----------

