# Databricks notebook source
# MAGIC %md
# MAGIC This problem involves writing functions.

# COMMAND ----------

# import statistical packages
import numpy as np
import pandas as pd

# COMMAND ----------

# import data visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC **a. Write a function, `Power()`, that prints out the result of raising 2
# MAGIC to the 3rd power. In other words, your function should compute
# MAGIC 23 and print out the results.**
# MAGIC
# MAGIC *Hint: Recall that `x**a` raises `x` to the power `a`. Use the print()
# MAGIC function to display the result.*

# COMMAND ----------

def power():
    print(pow(2, 3))

# COMMAND ----------

power()

# COMMAND ----------

# MAGIC %md
# MAGIC **b. Create a new function, `Power2()`, that allows you to pass any two numbers, x and a, and prints out the value of `x^a`. You can do this by beginning your function with the line**
# MAGIC <br>
# MAGIC `def Power2(x, a):`
# MAGIC <br>
# MAGIC <br>
# MAGIC You should be able to call your function by entering, for instance,
# MAGIC <br>
# MAGIC `Power2(3, 8)`

# COMMAND ----------

def power2(x, a):
    print(pow(x, a))

# COMMAND ----------

pow(3, 8)

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Using the `Power2()` function that you just wrote, compute 10^3, 8^17, and 131^3.**

# COMMAND ----------

power2(10, 3)

# COMMAND ----------

power2(8, 17)

# COMMAND ----------

power2(131, 3)

# COMMAND ----------

# MAGIC %md
# MAGIC **d. Now create a new function, `Power3()`, that actually returns the
# MAGIC result `x^a` as a `Python` object, rather than simply printing it
# MAGIC to the screen. That is, if you store the value `x^a` in an object
# MAGIC called result within your function, then you can simply return return
# MAGIC this result, using the following line:**
# MAGIC <br>
# MAGIC <br>
# MAGIC `return result`
# MAGIC <br>
# MAGIC <br>
# MAGIC Note that the line above should be the last line in your function.

# COMMAND ----------

def power3(x, a):
    return pow(x, a)

# COMMAND ----------

# MAGIC %md
# MAGIC **e. Now using the `Power3()` function, create a plot of f(x) = x2.
# MAGIC The x-axis should display a range of integers from 1 to 10, and
# MAGIC the y-axis should display x2. Label the axes appropriately, and
# MAGIC use an appropriate title for the fgure. Consider displaying either
# MAGIC the x-axis, the y-axis, or both on the log-scale. You can do this
# MAGIC by using the `ax.set_xscale()` and `ax.set_yscale()` methods of `.set_xscale()` and 
# MAGIC `.set_yscale()` the axes you are plotting to.**

# COMMAND ----------

def fx(x):
    return power3(x, 2)

# COMMAND ----------

df = pd.DataFrame()
for i in range(1,11):
    df = df.append([[i, fx(i)]], ignore_index = True)

# COMMAND ----------

df.columns = [['x', 'fx']]

# COMMAND ----------

df

# COMMAND ----------

plt.xkcd()
plt.figure(figsize = (25, 10))
plt.plot(df['fx'],color='green', linestyle='dashed', marker='o',
         markerfacecolor='yellow', markersize=10)
plt.title('fx vs x')
plt.xlabel('x')
plt.ylabel('fx = x^2')

# COMMAND ----------

# MAGIC %md
# MAGIC **f. Create a function, `PlotPower()`, that allows you to create a plot of `x` against `x^a` for a fixed `a` and a sequence of values of `x`. For  instance, if you call**
# MAGIC <br>
# MAGIC <br>
# MAGIC `PlotPower(np.arange(1, 11), 3)`
# MAGIC <br>
# MAGIC <br>
# MAGIC **then a plot should be created with an x-axis taking on values 1, 2,..., 10, and a y-axis taking on values 13, 23,..., 103.**

# COMMAND ----------

def PlotPower(x1, x2, a):
    df1 = pd.DataFrame()
    for i in range(x1, x2+1):
        df1 = df1.append([[i, power3 (i, a)]], ignore_index = True)
    df1.columns = [['x', 'x^a']]
    plt.xkcd()
    plt.figure(figsize = (25, 10))
    plt.plot(df1[['x^a']], color='green', linestyle='dashed', marker='o',
         markerfacecolor='yellow', markersize=10)
    plt.xticks(df1.index,df1["x"].values)
    plt.title('x^a vs x')
    plt.xlabel('x')
    plt.ylabel('x^a')
    return df1

# COMMAND ----------

PlotPower(12, 25, 3)