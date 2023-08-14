# Databricks notebook source
# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from scipy import stats
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,}'.format
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# import and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Wage.csv"
Wage = spark.read.option("header", "true").csv(url).toPandas()
Wage.set_index('SlNo', inplace=True)

int_cols = ["year", "age"]
float_cols = ["logwage", "wage"]
str_cols = ["maritl", "race", "education", "region", "jobclass", "health", "health_ins"]
Wage[int_cols] = Wage[int_cols].astype(int)
Wage[float_cols] = Wage[float_cols].astype(float)
Wage[str_cols] = Wage[str_cols].astype(str)

# COMMAND ----------

Wage.head()

# COMMAND ----------

Wage.describe().round(2)

# COMMAND ----------

Wage.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Polynomial regression

# COMMAND ----------

X1 = Wage['age']
X2 = X1**2
X3 = X1**3
X4 = X1**4
y = Wage['wage']
df4 = pd.concat([X1, X2, X3, X4], axis=1)

# COMMAND ----------

lmfit = ols('y~df4', data=Wage).fit()

# COMMAND ----------

lmsummary = lmfit.summary()
lmsummary.tables[1]

# COMMAND ----------

lmpred = lmfit.get_prediction(df4)
lmpred_df = lmpred.summary_frame()
lmpred_df.head().round(2)

# COMMAND ----------

lmse = pd.DataFrame([lmpred_df['mean']-2*lmpred_df['mean_se'], lmpred_df['mean']+2*lmpred_df['mean_se']]).T
lmse.columns = ['lower', 'upper']
lmse.head().round(2)

# COMMAND ----------

plotdf = pd.concat([y, X1, X2, X3, X4], axis=1)
plotdf.columns = ['wage', 'age', 'age^2', 'age^3', 'age^4']
plt.xkcd()
plt.figure(figsize = (25, 10))
sns.regplot('age', 'wage', data=plotdf, fit_reg=True, ci = 95, order=4, color='yellow', line_kws={'color':'green'})
plt.title('wage vs poly(age, 4)', fontsize=30)
plt.xlabel('poly(age, 4)', fontsize=20)
plt.ylabel('wage', fontsize=20)

# COMMAND ----------

df5 = pd.concat([plotdf, X1**5], axis=1)
df5.columns = ['wage', 'age', 'age^2', 'age^3', 'age^4', 'age^5']
df5.head().round(2)

# COMMAND ----------

lmfit1 = ols("df5['wage']~df5['age']", data=df5).fit()
lmfit2 = ols("df5['wage']~df5['age']+df5['age^2']", data=df5).fit()
lmfit3 = ols("df5['wage']~df5['age']+df5['age^2']+df5['age^3']", data=df5).fit()
lmfit4 = ols("df5['wage']~df5['age']+df5['age^2']+df5['age^3']+df5['age^4']", data=df5).fit()
lmfit5 = ols("df5['wage']~df5['age']+df5['age^2']+df5['age^3']+df5['age^4']+df5['age^5']", data=df5).fit()

# COMMAND ----------

anova_table = sm.stats.anova_lm(lmfit1, lmfit2, lmfit3, lmfit4, lmfit5, typ=1)
anova_table.index = anova_table.index+1
anova_table.round(6)

# COMMAND ----------

# MAGIC %md
# MAGIC *The p-value of the cubic and quartic models border on the statistical significance level of 5%, while the quintic model is unnecessary since its p-value>5%. Therfore, the ANOVA table suggets that a cubic or quartic model should suffice to explain the relationship between age and wage.*

# COMMAND ----------

# another way to explain the aforementioned phenomena
lmfit5.summary().tables[1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Polynomial logistic regression

# COMMAND ----------

df4 = df5.drop(columns='age^5')
df4['wage'] = np.where(df4['wage']>=250, 1, 0)

# COMMAND ----------

df4['wage'].value_counts()

# COMMAND ----------

df4.head()

# COMMAND ----------

logfit = sm.GLM(df4['wage'], df4[['age', 'age^2', 'age^3', 'age^4']], family=sm.families.Binomial()).fit()

# COMMAND ----------

logpred = logfit.get_prediction(df4[['age', 'age^2', 'age^3', 'age^4']])
logpred_df = logpred.summary_frame()
logpred_df.head()

# COMMAND ----------

logse = pd.DataFrame([logpred_df['mean']-2*logpred_df['mean_se'], logpred_df['mean']+2*logpred_df['mean_se']]).T
logse.columns = ['lower', 'upper']
logse.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step functions

# COMMAND ----------

df4_cut, bins = pd.cut(df4.age, 4, retbins = True, right = True)
df4_cut.value_counts()

# COMMAND ----------

lmfit_step = ols('df4.wage~df4_cut', data=df4).fit()
lmfit_step.summary()

# COMMAND ----------

