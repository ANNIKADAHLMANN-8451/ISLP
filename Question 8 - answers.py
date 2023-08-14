# Databricks notebook source
# MAGIC %md
# MAGIC This question involves the use of simple linear regression on the `Auto` data set.

# COMMAND ----------

# MAGIC %pip install --quiet chart-studio

# COMMAND ----------

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# COMMAND ----------

# import data visualisation tools
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Auto.csv"
Auto = spark.read.option("header", "true").csv(url).toPandas()

int_cols = ["cylinders", "horsepower", "weight", "year", "origin"]
float_cols = ["mpg", "displacement", "acceleration"]
str_cols = ["name"]
Auto[int_cols] = Auto[int_cols].astype(int)
Auto[float_cols] = Auto[float_cols].astype(float)
Auto[str_cols] = Auto[str_cols].astype(str)

# COMMAND ----------

Auto.head()

# COMMAND ----------

list(Auto)

# COMMAND ----------

Auto = Auto.drop(Auto.index[[32, 126, 330, 336, 354]]) # removing rows containing "?". This is the easy way out. Such missing values need to be explored first in a real life situation.

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Use the `sm.OLS()` function to perform a simple linear regression with `mpg` as the response and `horsepower` as the predictory. Use the `.summary()` function to print the results. Comment on the output. Consider using `stats.linregress` to gather the necessary information. For example:**
# MAGIC   - Is there a relationship between the predictor and the response?
# MAGIC   - How strong is the relationship between the predictor and the response?
# MAGIC   - Is the relationship between the predictor and the response positive or negative?
# MAGIC   - What is the predicted `mpg` associated with a `horsepower` of 98? What are the associated 95 % confdence and prediction intervals?

# COMMAND ----------

# run regression (I am not visualising the data for the sake of brevity. But it is highly recommended as first step afer importing data)
Y = Auto.mpg.astype(float)
X = Auto.horsepower.astype(float)
model = ols("Y ~ X", data = Auto).fit()

# COMMAND ----------

model.summary()

# COMMAND ----------

values = slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

# COMMAND ----------

X1 = Auto[['horsepower']]
Y1 = Auto['mpg']
lm = LinearRegression().fit(X1, Y1)

# COMMAND ----------

lm.coef_

# COMMAND ----------

print("Coefficients: ", lm.coef_)
print("Intercepts: ", lm.intercept_)
Y2 = [[98]]
predictions = lm.predict(Y2)

# COMMAND ----------

print("Predicted value at X = 98: ", predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC i. Given the F-Statistic > 1 and p-value of that F-Statistic is close to 0 (and << 0.005), there is a statistically significant
# MAGIC    relationship between mpg and horespower.
# MAGIC
# MAGIC
# MAGIC ii. To determine the strength of the relationship between the predictor (horsepower) and response (mpg), we need to calculate the ratio of the RSE of the predictor (as determined by the model) and the mean of the response, since it would show how strongly it actually predicts the true values of the response. (Another way would be to show R^2 since it would determine how much the regressed model actually explains the true values of the response). The RSE is 4.905757 and the mean of mpg is 23.445918, which means the percentage error is 20.923714%. The R^2 value is 0.795 or 79.5%.
# MAGIC
# MAGIC
# MAGIC iii. Negative relationship. An increase in horsepower is related to a decrease in mpg.
# MAGIC
# MAGIC iv. 24.46707715

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Plot the response and the predictory in a new `matplotlib` figure. Use the `sns.lmplot()` method to display the least squares regression line.**

# COMMAND ----------

# plot relationships
plt.xkcd()
plt.figure(figsize=(25, 10))
plotdata = pd.concat([X, Y], axis = 1)
sns.lmplot(x = "horsepower", y = "mpg", data = plotdata)
fig = plt.gcf()
fig.set_size_inches(25, 10)
plt.show()

# COMMAND ----------

