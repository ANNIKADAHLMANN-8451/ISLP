# Databricks notebook source
# MAGIC %md
# MAGIC We will now consider the `Boston` housing data set.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing

# COMMAND ----------

# import relevant statistical packages
import numpy as np
import pandas as pd
from statistics import stdev
from math import sqrt

# COMMAND ----------

# import relevant data visualisaton packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# load and preprocess data
url = "abfss://training@sa8451learningdev.dfs.core.windows.net/interpretable_machine_learning/eml_data/Boston.csv"
Boston = spark.read.option("header", "true").csv(url).toPandas()
Boston.set_index("SlNo", inplace=True)

int_cols = ['chas', 'rad']
float_cols = list(set(Boston.columns)-set(int_cols))
Boston[float_cols] = Boston[float_cols].astype(float)
Boston[int_cols] = Boston[int_cols].astype(int)

# COMMAND ----------

Boston.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Based on this data set, provide an estimate for the population
# MAGIC mean of `medv`. Call this estimate µˆ.**

# COMMAND ----------

mu_hat = round(Boston.medv.mean(), 2)
mu_hat

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Provide an estimate of the standard error of µˆ. Interpret this
# MAGIC result.**
# MAGIC <br>
# MAGIC *Hint: We can compute the standard error of the sample mean by
# MAGIC dividing the sample standard deviation by the square root of the
# MAGIC number of observations.*

# COMMAND ----------

sigma_hat = round(sqrt(Boston.medv.var()/(len(Boston.medv))), 4)
sigma_hat

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
sns.distplot(Boston.medv)

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Now estimate the standard error of µˆ using the bootstrap. How
# MAGIC does this compare to your answer from (b)?**

# COMMAND ----------

def meanfn(data, index):
    X = data.medv.iloc[index]
    return round(X.mean(), 2)

# COMMAND ----------

n = len(Boston.medv)
n

# COMMAND ----------

idx = list(range(n))

# COMMAND ----------

true_mean = meanfn(Boston, idx)
true_mean

# COMMAND ----------

from sklearn.utils import resample

# COMMAND ----------

estimate_mean = pd.DataFrame()
for i in range(n): # this is equivalent to constructing a new bootstrap data set and recomputing alpha_hat based on the new data set and generating 1,000 bootstrap estimates 
    bootstrap_estimate = meanfn(Boston, resample(idx, replace=True))
    estimate_mean = estimate_mean.append([bootstrap_estimate])    

# COMMAND ----------

estimate_mean.reset_index(drop=True, inplace=True)
estimate_mean.columns = ['Mean Estimate']
estimate_mean.head()

# COMMAND ----------

bootstrap_mean = round(estimate_mean['Mean Estimate'].mean(), 2)
bootstrap_mean

# COMMAND ----------

len(estimate_mean)

# COMMAND ----------

std_err = round(np.sqrt(np.sum(pow(estimate_mean - (np.sum(estimate_mean) / 506), 2))/505), 4) # check formula for standard error in page 189 of ISLR

# COMMAND ----------

std_err

# COMMAND ----------

# MAGIC %md
# MAGIC The bootstrap estimate of the mean is 22.55 with a standard error of 0.3939. This is not too dissimilar from the 
# MAGIC true (population) mean and standard error of 22.53 and 0.4089 respectively.

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Based on your bootstrap estimate from (c), provide a 95 % confdence interval for the mean of `medv`. Compare it to the results
# MAGIC obtained by using `Boston['medv'].std()` and the two standard
# MAGIC error rule (3.9).**
# MAGIC <br>
# MAGIC *Hint: You can approximate a 95 % confdence interval using the
# MAGIC formula [ˆµ − 2SE(ˆµ), µˆ + 2SE(ˆµ)].*

# COMMAND ----------

# MAGIC %md
# MAGIC Bootstrap estimates

# COMMAND ----------

conf_int_BS = [[round(bootstrap_mean - 2 * std_err, 4), round(bootstrap_mean + 2 * std_err, 4)]]
conf_int_BS

# COMMAND ----------

# MAGIC %md
# MAGIC Population estimates

# COMMAND ----------

conf_int_pop = [[round(mu_hat - 2 * sigma_hat, 4), round(mu_hat + 2 * sigma_hat, 4)]]
conf_int_pop

# COMMAND ----------

# MAGIC %md
# MAGIC The C.I. of the bootstrap estimates are narrower than the C.I. of the population estimates.

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Based on this data set, provide an estimate, µ_med, for the median
# MAGIC value of `medv` in the population.**

# COMMAND ----------

mu_med = Boston.medv.median()
mu_med

# COMMAND ----------

# MAGIC %md
# MAGIC **f. We now would like to estimate the standard error of µˆ_med. Unfortunately, there is no simple formula for computing the standard
# MAGIC error of the median. Instead, estimate the standard error of the
# MAGIC median using the bootstrap. Comment on your fndings.**

# COMMAND ----------

def medfn(data, index):
    X = data.medv.iloc[index]
    return round(X.median(), 2)

# COMMAND ----------

n = len(Boston.medv)
n

# COMMAND ----------

idx = list(range(n))

# COMMAND ----------

true_med = medfn(Boston, idx)
true_med

# COMMAND ----------

from sklearn.utils import resample

# COMMAND ----------

estimate_med = pd.DataFrame()
for i in range(n): # this is equivalent to constructing a new bootstrap data set and recomputing alpha_hat based on the new data set and generating 1,000 bootstrap estimates 
    bootstrap_estimate = medfn(Boston, resample(idx, replace=True))
    estimate_med = estimate_med.append([bootstrap_estimate])    

# COMMAND ----------

estimate_med.reset_index(drop=True, inplace=True)
estimate_med.columns = ['Median Estimate']
estimate_med.head()

# COMMAND ----------

std_err = round(np.sqrt(np.sum(pow(estimate_med - (np.sum(estimate_med) / 506), 2))/505), 4) # check formula for standard error in page 189 of ISLR

# COMMAND ----------

std_err

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the median and standard error using bootstrap are 21.2 and 0.3731 respectively. The standard error is small
# MAGIC relative to the the median.

# COMMAND ----------

# MAGIC %md
# MAGIC **g. Based on this data set, provide an estimate for the tenth percentile of `medv` in Boston census tracts. Call this quantity µˆ0.1.
# MAGIC (You can use the `np.percentile()` function.)**

# COMMAND ----------

mu_01 = Boston.medv.quantile(0.1)
mu_01

# COMMAND ----------

# MAGIC %md
# MAGIC **h. Use the bootstrap to estimate the standard error of µˆ0.1. Comment on your fndings.**

# COMMAND ----------

def perc01fn(data, index):
    X = data.medv.iloc[index]
    return round(X.quantile(0.1), 2)

# COMMAND ----------

n = len(Boston.medv)
n

# COMMAND ----------

idx = list(range(n))

# COMMAND ----------

true_perc01 = perc01fn(Boston, idx)
true_perc01

# COMMAND ----------

from sklearn.utils import resample

# COMMAND ----------

estimate_perc01 = pd.DataFrame()
for i in range(n): # this is equivalent to constructing a new bootstrap data set and recomputing alpha_hat based on the new data set and generating 1,000 bootstrap estimates 
    bootstrap_estimate = perc01fn(Boston, resample(idx, replace=True))
    estimate_perc01 = estimate_perc01.append([bootstrap_estimate])    

# COMMAND ----------

estimate_perc01.reset_index(drop=True, inplace=True)
estimate_perc01.columns = ['10th-percentile Estimate']
estimate_perc01.head()

# COMMAND ----------

std_err = round(np.sqrt(np.sum(pow(estimate_perc01 - (np.sum(estimate_perc01) / 506), 2))/505), 4) # check formula for standard error in page 189 of ISLR

# COMMAND ----------

std_err

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, the median and standard error using bootstrap are 12.75 and 0.4994 respectively. The standard error is small
# MAGIC relative to the the median.