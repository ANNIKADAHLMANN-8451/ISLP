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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Compute the cross-validation error rates for support vector
# MAGIC classifers with a range of C values. How many training observations are misclassifed for each value of C considered, and how
# MAGIC does this relate to the cross-validation errors obtained?**

# COMMAND ----------

from sklearn.model_selection import GridSearchCV as GSV
from sklearn.svm import SVC

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Generate an appropriate test data set, and compute the test
# MAGIC errors corresponding to each of the values of C considered. Which
# MAGIC value of C leads to the fewest test errors, and how does this
# MAGIC compare to the values of C that yield the fewest training errors
# MAGIC and the fewest cross-validation errors?**

# COMMAND ----------

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Discuss your results.**

# COMMAND ----------

# TODO: your response here