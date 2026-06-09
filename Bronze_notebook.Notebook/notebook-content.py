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

# %pip install prince

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### steps
#     > clean the dataset
#     > statistical test
#     > encode
#     > Split the data (There is another dataset to test but split this for training and validation)
#     > deal with the imbalance dataset 
#     > normalize / scale 
#     > train and validate 
#     > Evaluate
#     > Test on the test dataset

# CELL ********************

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
import pyspark
from pyspark.sql.types import *
from pyspark.sql import functions as f
from scipy import stats
import itertools
import seaborn as sns
from sklearn.decomposition import PCA
#from prince import FAMD

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# age             
# job            
# marital       
# education     
# default       
# housing       
# loan          
# contact       
# month         
# day_of_week  Last contacted 
# 
# duration  last contact duration (delete it before model training) 
# 
# campaign - number of contacts during this campaign  
# 
# pday  - Days since client was last contacted (999 means never) -   Does it mean that the customer has subscribed without been contacted to the term deposit?     
# 
# previous  - number of contacts before this campaign 
# 
# poutcome  - outcome of previous campaign - needs to be hot encoded -also think about 
# 
# emp.var.rate - Employment variation rate +1.4 good, -3.4 bad 
# 
# cons.price.idx - consumer price index 
# 
# cons.conf.idx - consumer confidence index
# 
# euribor3m - Euribor 3 month rate - the interest rate at which European banks lend to each other
# 
# nr.employed  - number of employees in the market - the higher the better
# 
# y     - target variable shows whether the customer is subscribed or not. - have to consider the class imbalance 


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

# Convert the CSV file to delta table 

#df.write.format('delta').option('overwriteSchema', 'false').saveAsTable('silver_bank_market')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Read the data in to spark dataframe

df = spark.read.format('delta').load('abfss://project1_bank_market@onelake.dfs.fabric.microsoft.com/project1_lakehouse.Lakehouse/Tables/dbo/silver_bank_market')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Converts the data to pandas df
p_df = df.toPandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Basic expoleration

p_df.info() 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# basic expoleration
p_df.describe().round(2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Find the unique categories in the target variable (it is a binary)

p_df.y.unique()
p_df['y'] = p_df['y'].map({'no':0, 'yes':1})# Encode the targeted variable


# Select the category and numerical columns and save them in a df

#categorical cols
unique_category = p_df.select_dtypes(include=object).columns
cat_col_df = p_df[unique_category].copy()
cat_col_df

# numerical value columns - df
num_col = p_df.select_dtypes(include=[int,float]).columns
num_col_df = p_df[num_col].copy()
num_col_df

cat_col_df.shape, num_col_df.shape

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Statistical analysis
# 
# 1. Check if the independent variables are statistical independent
# 2. Correlation between the dependent variable 
# if there is dependency between - think about feature extraction 


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

# numerical col distribution

file_path = '/lakehouse/default/Files/plots/hist.png'

#num_col_df.hist(bins=25, figsize=(15,10))
#plt.savefig(file_path, dpi=300)
#plt.close()

#display the saved image

display(Image(filename=file_path))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# use spareman correlation measure with 98% confidence interval

corr_num = num_col_df.corr(method='spearman')
corr_num.round(3)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

#     number of employed and employment variation rate correlate to each other and have similar negative correlation between y. exclude the number of employed variable from the training.
# 
#     Duration column should also be excluded from the model training.

# CELL ********************

# drop the nr.employed column 

num_col_df_train = num_col_df.drop(['nr.employed','duration'], axis=1)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# def correlation(df):
#     stat_list=[]
#     for x in df.columns:
#         for y in df.columns:
#             res = stats.spearmanr(df[x],df[y])
#             if res.pvalue <= 0.05:
#                 stat_list.append({
#                     'column_1': x,
#                     'cloumn_2': y,
#                     'correlation': res.statistic,
#                     'p_value': res.pvalue,
#                     'relationship': 'yes'

#             })
            
#             else:
#                  stat_list.append({
#                     'column_1': x,
#                     'cloumn_2': y,
#                     'correlation': res.statistic,
#                     'p_value': res.pvalue,
#                     'relationship': 'no'
#                  }
                    
#                 )

#     return stat_list 


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

result = num_col_df_train.corr(method='spearman').round(3)

# Set the results in a df
stats_result_df = pd.DataFrame(result)

stats_result_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# plot the correlation df

sns.heatmap(result)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

num_train_spark = spark.createDataFrame(num_col_df_train) 
num_train_spark.write.format('delta').mode('overwrite').save('abfss://project1_bank_market@onelake.dfs.fabric.microsoft.com/project1_lakehouse.Lakehouse/Tables/dbo/num_table_train')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### Categorical columns

# CELL ********************

# visualize the unique categories in each column, count the values in each category

for col in cat_col_df:

    plt.figure(figsize=(10,6))
    cat_col_df[col].value_counts().plot(kind='bar')
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

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
