# Databricks notebook source
# MAGIC %md
# MAGIC This problem involves the `OJ` data set.

# COMMAND ----------

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

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/OJ.csv"
df = spark.read.option("header", "true").csv(url).toPandas()
df.set_index("SlNo", inplace=True)

str_cols = ["Purchase", "Store7"]
float_cols = ["PriceCH", "PriceMM", "DiscCH", "DiscMM", "LoyalCH", "SalePriceMM", "SalePriceCH", "PriceDiff", "PctDiscMM", "PctDiscCH", "ListPriceDiff"]
int_cols = list(set(df.columns)-set(str_cols)-set(float_cols))
df[str_cols] = df[str_cols].astype(str)
df[float_cols] = df[float_cols].astype(float)
df[int_cols] = df[int_cols].astype(int)

# COMMAND ----------

df.head()

# COMMAND ----------

df.Purchase = df.Purchase.map(lambda x: 1 if x=='CH' else 0)
df.Store7 = df.Store7.map({'No': 0, 'Yes': 1})

# COMMAND ----------

df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.**

# COMMAND ----------

X = df.drop(columns='Purchase') # For this, I checked question 9.b. to find out the response variable
y = df.Purchase

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.747663551402, test_size=0.25233644859, random_state=42)

# COMMAND ----------

len(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Fit a tree to the training data, with `Purchase` as the response
# MAGIC and the other variables as predictors. What is the training error
# MAGIC rate?**

# COMMAND ----------

tree = DecisionTreeClassifier(max_depth = 6).fit(X_train, y_train)
tree_score = tree.score(X_train, y_train)
print("Training error rate: ", round(1-tree_score, 2))

# COMMAND ----------

tree.get_params()

# COMMAND ----------

tree_pred = tree.predict(X_test)
class_mat = pd.DataFrame(confusion_matrix(y_test, tree_pred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(class_mat)

# COMMAND ----------

print(classification_report(y_test, tree_pred))

# COMMAND ----------

plot_tree(tree)

# COMMAND ----------

# MAGIC %md
# MAGIC The training error is approximately 0.11

# COMMAND ----------

# MAGIC %md
# MAGIC **c. How many
# MAGIC terminal nodes does the tree have?**

# COMMAND ----------

# MAGIC %md
# MAGIC 42 leave nodes

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Use cross-validation on the training set in order to determine
# MAGIC the optimal tree size.**

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

# MAGIC %md
# MAGIC Therefore, a tree at $m$=10 leaves has the lowest cross-validated classification error rate.

# COMMAND ----------

# MAGIC %md
# MAGIC **i. Produce a pruned tree corresponding to the optimal tree size
# MAGIC obtained using cross-validation. If cross-validation does not lead
# MAGIC to selection of a pruned tree, then create a pruned tree with fve
# MAGIC terminal nodes.**

# COMMAND ----------

pruned_tree = DecisionTreeClassifier(max_depth = 10)
pruned_tree.fit(X_train, y_train)
pruned_tree_score = pruned_tree.score(X_train, y_train)
print("Training error rate: ", round(1-pruned_tree_score, 2))

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, non-pruned tree results in higher training error rate.

# COMMAND ----------

pruned_tree_pred = pruned_tree.predict(X_test)
pruned_class_mat = pd.DataFrame(confusion_matrix(y_test, pruned_tree_pred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(pruned_class_mat)

# COMMAND ----------

print(classification_report(y_test, pruned_tree_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, pruned tree results in higher test error rate.