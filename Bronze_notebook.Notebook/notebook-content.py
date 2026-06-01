# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "851bb40d-4e51-46aa-9599-c2a5abf1cd1c",
# META       "default_lakehouse_name": "project1_lakehouse",
# META       "default_lakehouse_workspace_id": "c8ec0e1c-8e68-428f-893e-3e37686b2386",
# META       "known_lakehouses": [
# META         {
# META           "id": "851bb40d-4e51-46aa-9599-c2a5abf1cd1c"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

%pip install prince

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql.types import *
from pyspark.sql import functions as f
from scipy import stats
import itertools
import seaborn as sns
from sklearn.decomposition import PCA
import prince

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create a schema structure
schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("marital", StringType(), True),
    StructField("education", StringType(), True),
    StructField("default", StringType(), True),
    StructField("housing", StringType(), True),
    StructField("loan",StringType(), True),
    StructField("contact", StringType(), True),
    StructField("month", StringType(), True),
    StructField("day_of_week", StringType(), True),
    StructField("duration",IntegerType(), True),
    StructField("campaign", IntegerType(), True),
    StructField("pday", IntegerType(), True),
    StructField("previous", IntegerType(), True),
    StructField("poutcome", StringType(), True),
    StructField("emp.var.rate",FloatType(), True),
    StructField("cons.price.idx",FloatType(), True),
    StructField("cons.conf.idx",FloatType(), True),
    StructField("euribor3m", FloatType(), True),
    StructField("nr.employed", FloatType(), True),
    StructField("y", StringType(), True)   
])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.read.format("csv").schema(schema=schema).option("header","true").option("delimiter",";").load("Files/bronze_layer/bank-additional-full.csv")
# df now is a Spark DataFrame containing CSV data from "Files/bronze_layer/bank-additional-full.csv".
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#df.write.format('delta').option('overwriteSchema', 'false').saveAsTable('silver_bank_market')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Null enteries

df = spark.read.format('delta').load('abfss://project1_bank_market@onelake.dfs.fabric.microsoft.com/project1_lakehouse.Lakehouse/Tables/dbo/silver_bank_market')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

p_df = df.toPandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# check for null values
p_df.isnull().sum(), p_df.isna().sum()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

p_df.describe().round(2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Statistical analysis
# 
# 1. Check if the independent variables are statistical independent
# 2. correlation between the dependent variable 
# if there is dependency between - think about feature extraction 


# CELL ********************

# numerical value columns - df
num_col = p_df.select_dtypes(include=[int,float]).columns
num_col_df = p_df[num_col].copy()
num_col_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create a box plot to check for outlairs

for col in num_col_df:
    plt.figure(figsize=(10,6))
    num_col_df[col].plot(kind='box')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Categorical columns
cat_col = p_df.select_dtypes(include=object).columns
cat_col_df = p_df[cat_col].copy()
cat_col_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# use spareman correlation measure with 98% confidence interval

corr_num = num_col_df.corr(method='spearman')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

corr_num.round(2)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def correlation(df):
    stat_list=[]
    for x in df.columns:
        for y in df.columns:
            res = stats.spearmanr(df[x],df[y])
            if res.pvalue <= 0.05:
                stat_list.append({
                    'column_1': x,
                    'cloumn_2': y,
                    'correlation': res.statistic,
                    'p_value': res.pvalue,
                    'relationship': 'yes'

            })
            
            else:
                 stat_list.append({
                    'column_1': x,
                    'cloumn_2': y,
                    'correlation': res.statistic,
                    'p_value': res.pvalue,
                    'relationship': 'no'
                 }
                    
                )

    return stat_list


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

result = correlation(num_col_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Set the results in a df
stats_result_df = pd.DataFrame(result)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

stats_result_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# for ploting
cor_plot = num_col_df.corr(method='spearman')

sns.heatmap(
    cor_plot

)

please upload to lakehouse

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Select the category columns
unique_category = df.select_dtypes(include=object).columns

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# visualize the unique categories in each column, count the values in each category, cal outof the contacted customers 

for col in unique_category:
    plt.figure(figsize=(10,6))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
