# Databricks notebook source
# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

# COMMAND ----------

# import data visualisation packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Carseats.csv"
df = spark.read.option("header", "true").csv(url).toPandas()
df.set_index("SlNo")

int_cols = ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]
float_cols = ["Sales"]
str_cols = ["ShelveLoc", "Urban", "US"]
df[int_cols] = df[int_cols].astype(int)
df[float_cols] = df[float_cols].astype(float)
df[str_cols] = df[str_cols].astype(str)

df.Sales = df.Sales.map(lambda x: 0 if x<=8 else 1)
df.ShelveLoc = pd.factorize(df.ShelveLoc)[0]
df.Urban = df.Urban.map({'No':0, 'Yes':1})
df.US = df.US.map({'No':0, 'Yes':1})

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performing decision tree classification

# COMMAND ----------

X = df.drop(['Sales'], axis = 1)
y = df.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, random_state = 0)

# COMMAND ----------

tree_carseats = DecisionTreeClassifier(max_depth = 2)
tree_carseats.fit(X_train, y_train)
tree_carseats.score(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC **This suggests that the training accuracy of the model is 74%. However, the true test of the model's predictive ability lies in the test set.**

# COMMAND ----------

tree_pred = tree_carseats.predict(X_test)
class_mat = pd.DataFrame(confusion_matrix(y_test, tree_pred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(class_mat)

# COMMAND ----------

print(classification_report(y_test, tree_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC **This suggests that the model predicts 69% of correct predictions in the test set.**

# COMMAND ----------

plot_tree(tree_carseats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pruning the decision tree

# COMMAND ----------

SCORES = []
max_leafs_arr = range(2, 50)
for max_leafs in max_leafs_arr:
    regressionTree = DecisionTreeClassifier(max_leaf_nodes=max_leafs)
    sc = cross_val_score(regressionTree, X, y, cv=10, scoring="neg_mean_squared_error")
    SCORES.append((-sc.mean(), sc.std()))
SCORES = np.array(SCORES)

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
plt.plot(max_leafs_arr, SCORES[:,0], 'g')
plt.fill_between(max_leafs_arr, SCORES[:,0]+SCORES[:,1], SCORES[:,0]-SCORES[:,1], alpha=0.3, color='y')
plt.xlabel('tree size', fontsize=20, color='c')
plt.ylabel('MSE', fontsize=20, color='c')
plt.title('finding the best tree through cross-validation', fontsize=30, color='m')
best_min_leafs = max_leafs_arr[np.argmin(SCORES[:,0])]
print(f"The best tree has {best_min_leafs} leafs.")

# COMMAND ----------

tree_carseats_1 = DecisionTreeClassifier(max_depth = 17)
tree_carseats_1.fit(X_train, y_train)
tree_carseats_1.score(X_train, y_train)

# COMMAND ----------

tree_pred_1 = tree_carseats_1.predict(X_test)
class_mat_1 = pd.DataFrame(confusion_matrix(y_test, tree_pred_1).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(class_mat_1)

# COMMAND ----------

print(classification_report(y_test, tree_pred_1))

# COMMAND ----------

plot_tree(tree_carseats_1)

# COMMAND ----------

# MAGIC %md
# MAGIC **I can achieve the closest to 38 leaves when depth = 17. At 17 leaves with pruning, the predictive ability of the decision tree increases to 72%, which succesfully demonstrates how pruning via. cross-validation can be useful).**