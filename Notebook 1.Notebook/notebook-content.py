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

# Welcome to your new notebook
# Type here in the cell editor to add code!


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import matplotlib.pyplot as plt

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Read the dataset 
df = pd.read_csv('abfss://project1_bank_market@onelake.dfs.fabric.microsoft.com/project1_lakehouse.Lakehouse/Files/bronze_layer/bank-additional-full.csv',delimiter=';')
df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Null enteries

df.info()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df.describe()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Select the category columns
unique_category = df.select_dtypes(include=object).columns

#Get the unique categories in each column, count the values in each category, cal outof the contacted customers 

for col in unique_category:
    
    print('********')
    print(f'{col}: {df[col].nunique()}\n {df[col].value_counts()}')
    percentage = df[col].value_counts(normalize=True)*100
    counts = df[col].value_counts()
    sum_df = pd.DataFrame({'Counts': counts, 'Percentage':percentage.round(2)})
print()
print(sum_df)
   
   


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The duration variable should be discarded while training the classifier model.
# The dependent variable is y

# CELL ********************

# 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
