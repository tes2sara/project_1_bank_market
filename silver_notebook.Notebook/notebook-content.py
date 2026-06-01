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

import pandas as pd
import pyspark
from pyspark.sql.types import * 
from pyspark.sql.functions import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.read.format('delta').option('header','true').load('abfss://project1_bank_market@onelake.dfs.fabric.microsoft.com/project1_lakehouse.Lakehouse/Tables/dbo/silver_bank_market')
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
