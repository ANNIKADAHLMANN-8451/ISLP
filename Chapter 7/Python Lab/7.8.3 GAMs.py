# Databricks notebook source
# MAGIC %pip install statsmodels==0.13.2 pygam

# COMMAND ----------

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
# MAGIC ### GAMs

# COMMAND ----------

from pygam import LinearGAM, LogisticGAM, s, f
from patsy import dmatrix

# COMMAND ----------

X = pd.concat([Wage['year'], Wage['age'], Wage['education'].astype('category').cat.codes], axis=1)
X.rename(columns={0: 'education'}, inplace=True)
y = Wage['wage']

# COMMAND ----------

X.head()

# COMMAND ----------

X_age = dmatrix("cr(AGE, df=5)", {"AGE": Wage['age']}, return_type='dataframe')
X_year = dmatrix("cr(YEAR, df=4)", {"YEAR": Wage['year']}, return_type='dataframe')
X_education = Wage['education']
y = Wage.wage
df = pd.concat([y, X_year, X_age, X_education], axis=1)
df.head()

# COMMAND ----------

gam1 = ols('y~X_year+X_age+X_education', data=df).fit()
gam1.summary()

# COMMAND ----------

X_age_1 = dmatrix("cr(AGE, df=5)", {"AGE": Wage['age']}, return_type='dataframe')
X_year_1 = dmatrix("cr(YEAR, df=4)", {"YEAR": Wage['year']}, return_type='dataframe')
X_education_1 = Wage['education']
y_1 = Wage.wage
df_1 = pd.concat([y_1, X_year_1, X_age_1, X_education_1], axis=1)
df_1.head()

# COMMAND ----------

gam_1 = ols('y~X_year', data=df).fit()
gam_2 = ols('y~X_year+X_age', data=df).fit()
gam_3 = ols('y~X_year+X_age+X_education', data=df).fit() 

# COMMAND ----------

anova_table = sm.stats.anova_lm(gam_1, gam_2, gam_3, typ=1)
anova_table.index = anova_table.index+1
anova_table.round(6)

# COMMAND ----------

gam_3.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **Logistic regression GAM**

# COMMAND ----------

X_age = dmatrix("cr(AGE, df=5)", {"AGE": Wage['age']}, return_type='dataframe')
X_year = Wage['year']
X_education = Wage['education'].astype('category').cat.codes
y_wage = np.where(Wage['wage']>=250, 1, 0)
dflog = np.array(pd.concat([X_age, X_year, X_education], axis=1))

# COMMAND ----------

glmlog = sm.GLM(y_wage, dflog, family=sm.families.Binomial()).fit()
glmlog.summary()

# COMMAND ----------

