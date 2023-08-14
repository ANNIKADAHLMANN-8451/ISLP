# Databricks notebook source
# MAGIC %md
# MAGIC At the end of Section 9.6.1, it is claimed that in the case of data that is
# MAGIC just barely linearly separable, a support vector classifer with a small
# MAGIC value of C that misclassifes a couple of training observations may
# MAGIC perform better on test data than one with a huge value of C that does
# MAGIC not misclassify any training observations. You will now investigate
# MAGIC this claim

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
# MAGIC **a. Generate two-class data with p = 2 in such a way that the classes
# MAGIC are just barely linearly separable.**

# COMMAND ----------

x1 = np.random.uniform(low=0.0, high=90.0, size=500)
y1 = np.random.uniform(low=x1+10, high=100.0, size=500)
x1_noise = np.random.uniform(low=20.0, high=80.0, size=50)
y1_noise = (5/4) + (x1_noise - 10) + 0.1

# COMMAND ----------

x0 = np.random.uniform(low=10.0, high=100.0, size=500)
y0 = np.random.uniform(low=0.0, high=x0-10, size=500)
x0_noise = np.random.uniform(low=20.0, high=80.0, size=50)
y0_noise = (5/4) + (x0_noise - 10) - 0.1

# COMMAND ----------

class1 = range(0,551)
class2 = range(551, 1100)

# COMMAND ----------

X = pd.concat([pd.DataFrame([x1]), pd.DataFrame([x1_noise]), pd.DataFrame([x0]), pd.DataFrame([x0_noise])], axis=1).T
X.columns = ['X']
X.head()

# COMMAND ----------

Y = pd.concat([pd.DataFrame([y1]), pd.DataFrame([y1_noise]), pd.DataFrame([y0]), pd.DataFrame([y0_noise])], axis=1).T
Y.columns = ['Y']
Y.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(X.iloc[class1], Y.iloc[class1], s=250, alpha=0.65, cmap='viridis')
plt.scatter(X.iloc[class2], Y.iloc[class2], s=250, alpha=0.65, cmap='viridis')
plt.title('sample data with barely linearly separable classes', color='m', fontsize=30)

# COMMAND ----------

# MAGIC %md
# MAGIC This plot creates a barely separable linear boundary at $5x - 4y - 50 = 0.$

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Compute the cross-validation error rates for support vector
# MAGIC classifers with a range of C values. How many training observations are misclassifed for each value of C considered, and how
# MAGIC does this relate to the cross-validation errors obtained?**

# COMMAND ----------

from sklearn.model_selection import GridSearchCV as GSV
from sklearn.svm import SVC

# COMMAND ----------

Z = pd.DataFrame([np.zeros(shape=1100, dtype=int)]).T
Z.iloc[class1] = 1
Z.columns = ['Z']
Z.head()

# COMMAND ----------

df = pd.concat([X, Y], axis=1)
df.head()

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
cost_range = [{'C': [0.01, 0.1, 1, 5, 10, 100, 1000, 10000], 'gamma': [0.5, 1,2,3,4, 5, 6]}]
cost_cv= GSV(SVC(kernel='rbf'), cost_range, cv=10, scoring='accuracy', return_train_score=True).fit(df, Z.Z)

# COMMAND ----------

best_params = cost_cv.best_params_
best_params['C']

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# COMMAND ----------

means = pd.DataFrame([cost_cv.cv_results_['mean_test_score']]).T
means.columns = ['means']
means.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(means, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('mean test score', fontsize=30, color='m')

# COMMAND ----------

std = pd.DataFrame([cost_cv.cv_results_['std_test_score']]).T
std.columns = ['std']
std.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(std, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('std test score', fontsize=30, color='m')

# COMMAND ----------

resultsdf = pd.concat([means, std], axis=1)
resultsdf.head()

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

C = [0.01, 0.1, 1, 5, 10, 100, 1000, 10000]
g = [0.5, 1, 2,3,4, 5, 6]

# COMMAND ----------

for k in C:
    print(g)

# COMMAND ----------

errdf = pd.DataFrame()
for k in C:
    for p in g:
        svmfit = SVC(C=k, kernel='rbf', gamma=p).fit(df, Z)
        Zpred = svmfit.predict(df)
        class_mat = pd.DataFrame(confusion_matrix(Z, Zpred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
        err_perc = (class_mat.No.No + class_mat.Yes.Yes)/1100 * 100
        errdf = errdf.append(pd.DataFrame([err_perc]))

# COMMAND ----------

errdf.reset_index(drop=True, inplace=True)
errdf.columns= ['errors']
plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(100-errdf, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('number of misclassifications', fontsize=30, color='m')

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Generate an appropriate test data set, and compute the test
# MAGIC errors corresponding to each of the values of C considered. Which
# MAGIC value of C leads to the fewest test errors, and how does this
# MAGIC compare to the values of C that yield the fewest training errors
# MAGIC and the fewest cross-validation errors?**

# COMMAND ----------

X_test = np.random.uniform(low=0.0, high=100.0, size=1000)
class_1 = np.random.randint(low=0, high=1000, size=500)
y_test = np.zeros(1000)

# COMMAND ----------

x1_test = np.random.uniform(low=0.0, high=90.0, size=500)
y1_test = np.random.uniform(low=x1_test+10, high=100.0, size=500)
x1_noise_test = np.random.uniform(low=20.0, high=80.0, size=50)
y1_noise_test = (5/4) + (x1_noise_test - 10) + 0.1

# COMMAND ----------

x0_test = np.random.uniform(low=10.0, high=100.0, size=500)
y0_test = np.random.uniform(low=0.0, high=x0-10, size=500)
x0_noise_test = np.random.uniform(low=20.0, high=80.0, size=50)
y0_noise_test = (5/4) + (x0_noise_test - 10) - 0.1

# COMMAND ----------

class1 = range(0,551)
class2 = range(551, 1100)

# COMMAND ----------

X = pd.concat([pd.DataFrame([x1_test]), pd.DataFrame([x1_noise_test]), pd.DataFrame([x0_test]), pd.DataFrame([x0_noise_test])], axis=1).T
X.columns = ['X_test']
X.head()

# COMMAND ----------

Y = pd.concat([pd.DataFrame([y1_test]), pd.DataFrame([y1_noise_test]), pd.DataFrame([y0_test]), pd.DataFrame([y0_noise_test])], axis=1).T
Y.columns = ['Y_test']
Y.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.scatter(X.iloc[class1], Y.iloc[class1], s=250, alpha=0.65, cmap='viridis')
plt.scatter(X.iloc[class2], Y.iloc[class2], s=250, alpha=0.65, cmap='viridis')
plt.title('test data', color='m', fontsize=30)

# COMMAND ----------

Z = pd.DataFrame([np.zeros(shape=1100, dtype=int)]).T
Z.iloc[class1] = 1
Z.columns = ['Z']
Z.head()

# COMMAND ----------

df = pd.concat([X, Y], axis=1)
df.head()

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
cost_range = [{'C': [0.01, 0.1, 1, 5, 10, 100, 1000, 10000], 'gamma': [0.5, 1,2,3,4, 5, 6]}]
cost_cv= GSV(SVC(kernel='rbf'), cost_range, cv=10, scoring='accuracy', return_train_score=True).fit(df, Z.Z)

# COMMAND ----------

best_params = cost_cv.best_params_
best_params['C']

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# COMMAND ----------

means = pd.DataFrame([cost_cv.cv_results_['mean_test_score']]).T
means.columns = ['means']
means.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(means, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('mean test score', fontsize=30, color='m')

# COMMAND ----------

std = pd.DataFrame([cost_cv.cv_results_['std_test_score']]).T
std.columns = ['std']
std.head()

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(std, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('std test score', fontsize=30, color='m')

# COMMAND ----------

resultsdf = pd.concat([means, std], axis=1)
resultsdf.head()

# COMMAND ----------

from sklearn.metrics import confusion_matrix, classification_report

# COMMAND ----------

C = [0.01, 0.1, 1, 5, 10, 100, 1000, 10000]
g = [0.5, 1, 2,3,4, 5, 6]

# COMMAND ----------

for k in C:
    print(g)

# COMMAND ----------

errdf = pd.DataFrame()
for k in C:
    for p in g:
        svmfit = SVC(C=k, kernel='rbf', gamma=p).fit(df, Z)
        Zpred = svmfit.predict(df)
        class_mat = pd.DataFrame(confusion_matrix(Z, Zpred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
        err_perc = (class_mat.No.No + class_mat.Yes.Yes)/1100 * 100
        errdf = errdf.append(pd.DataFrame([err_perc]))

# COMMAND ----------

errdf.reset_index(drop=True, inplace=True)
errdf.columns= ['errors']
plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(100-errdf, c='g', ls='-.', marker='o', markerfacecolor='orange')
plt.title('number of misclassifications', fontsize=30, color='m')

# COMMAND ----------

# MAGIC %md
# MAGIC $cost$=5 seems to be performing best on test data while $cost$=10 seems to perform best on training data.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Discuss your results.**

# COMMAND ----------

# MAGIC %md
# MAGIC A large cost results in overfitting and greater misclassification as opposed to a smaller cost.