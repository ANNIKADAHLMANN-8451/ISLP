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

# TODO: your response here

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

# TODO: your response here

# COMMAND ----------

# MAGIC %md
# MAGIC **c. Using the `Power2()` function that you just wrote, compute 10^3, 8^17, and 131^3.**

# COMMAND ----------

# TODO: your response here

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

# TODO: your response here

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

# TODO: your response here

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

# TODO: your response here