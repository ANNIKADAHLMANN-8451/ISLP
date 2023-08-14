# Databricks notebook source
# MAGIC %md
# MAGIC The `Wage` data set contains a number of other features not explored
# MAGIC in this chapter, such as marital status (`maritl`), job class (`jobclass`),
# MAGIC and others. Explore the relationships between some of these other
# MAGIC predictors and `wage`, and use non-linear ftting techniques in order to
# MAGIC ft fexible models to the data. Create plots of the results obtained,
# MAGIC and write a summary of your fndings.

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing

# COMMAND ----------

# import packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,}'.format
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Wage.csv"
Wage = spark.read.option("header", "true").csv(url).toPandas()
Wage.set_index("SlNo", inplace=True)

str_cols = ["maritl", "race", "education", "region", "jobclass", "health", "health_ins"]
float_cols = ["logwage", "wage"]
int_cols = list(set(Wage.columns)-set(str_cols)-set(float_cols))
Wage[str_cols] = Wage[str_cols].astype(str)
Wage[float_cols] = Wage[float_cols].astype(float)
Wage[int_cols] = Wage[int_cols].astype(int)

# COMMAND ----------

Wage.head()

# COMMAND ----------

Wage.describe().round(2)

# COMMAND ----------

Wage.info()

# COMMAND ----------

# MAGIC %md
# MAGIC Exploring relationships with other features in the Wage data set

# COMMAND ----------

# MAGIC %md
# MAGIC 'maritl' and 'jobclass'

# COMMAND ----------

# MAGIC %md
# MAGIC *Basic exploration of the dataset*

# COMMAND ----------

Wage.maritl.value_counts(sort=True)

# COMMAND ----------

Wage.jobclass.value_counts(sort=True)

# COMMAND ----------

plt.xkcd()
fig, axes = plt.subplots(1, 2, figsize=(25,10))

sns.boxplot(Wage.maritl, Wage.wage, ax=axes[0])
axes[0].set_xlabel('maritl', fontsize=20, color='c')
axes[0].set_ylabel('wage', fontsize=20, color='c')
axes[0].set_title('wage as function of martial status', color='m', fontsize=30)


sns.boxplot(Wage.jobclass, Wage.wage, ax=axes[1])
axes[1].set_xlabel('jobclass', fontsize=20, color='c')
axes[1].set_ylabel('wage', fontsize=20, color='c')
axes[1].set_title('wage as function of jobclass', color='m', fontsize=30)

# COMMAND ----------

# MAGIC %md
# MAGIC Initial plots show married people and those in the the information sector make more money than their counterparts.

# COMMAND ----------

# MAGIC %md
# MAGIC Polynomial regression

# COMMAND ----------

# MAGIC %md
# MAGIC *I cannot increase degrees of categorical variables like 'maritl' and 'jobclass'. Therefore, it will reduce to normal linear regression.*

# COMMAND ----------

X1 = Wage.maritl.astype('category').cat.codes
X2 = Wage.jobclass.astype('category').cat.codes
y = Wage.wage
df = pd.concat([y, X1, X2], axis=1)
df.columns = ['wage', 'maritl', 'jobclass']
df.head()

# COMMAND ----------

lm1 = ols('df.wage~df.maritl', data=df).fit()
lm1.summary()

# COMMAND ----------

lm2 = ols('df.wage~df.jobclass', data=df).fit()
lm2.summary()

# COMMAND ----------

lm3 = ols('df.wage~df.maritl+df.jobclass', data=df).fit()
lm3.summary()

# COMMAND ----------

MSE_df = pd.concat([pd.DataFrame([lm1.mse_model]), pd.DataFrame([lm2.mse_model]), pd.DataFrame([lm3.mse_model])], axis=1)
MSE_df.columns = ['lm1', 'lm2', 'lm3']
MSE_df = MSE_df.T
MSE_df.columns = ['Model MSE']
MSE_df

# COMMAND ----------

# MAGIC %md
# MAGIC The model with marital status as the sole regressor provides the least MSE.

# COMMAND ----------

# MAGIC %md
# MAGIC Splines

# COMMAND ----------

# MAGIC %md
# MAGIC *Splines cannot be fit on categorical variables.*

# COMMAND ----------

# MAGIC %md
# MAGIC GAM

# COMMAND ----------

from patsy import dmatrix
X3 = Wage.age
X3_age = dmatrix("cr(AGE, df=4)", {"AGE": Wage['age']}, return_type='dataframe')
df = pd.concat([df, X3_age], axis=1)
df.head()

# COMMAND ----------

lm_gam = ols('df.wage~df.maritl+df.jobclass+X3_age', data=df).fit()
lm_gam.summary()

# COMMAND ----------

lmgam_mse = lm_gam.mse_model

# COMMAND ----------

MSE_df = pd.concat([pd.DataFrame([lm1.mse_model]), pd.DataFrame([lm2.mse_model]), pd.DataFrame([lm3.mse_model]), pd.DataFrame([lmgam_mse])], axis=1)
MSE_df.columns = ["regression(maritl)", "regression(jobclass)", "regression(maritl+jobclass)", 'GAM']
MSE_df = MSE_df.T
MSE_df.columns = ['model_MSE']
MSE_df

# COMMAND ----------

plt.xkcd()
plt.figure(figsize=(25, 10))
sns.barplot(x=MSE_df.index, y=MSE_df.model_MSE, data=MSE_df)
plt.xlabel('models', fontsize=20, color='c')
plt.ylabel('model_MSE', fontsize=20, color='c')
plt.title('model MSE for different models', fontsize=30, color='m')