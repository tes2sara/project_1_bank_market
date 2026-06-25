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
#     > stratify test
#     > Split the data (There is another dataset to test but split this for training and validation)
#     > deal with the imbalance dataset 
#     > normalize / scale 
#     > train and validate 
#     > Evaluate
#     > Test on the test dataset

# CELL ********************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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

#p_df.y.unique()
#p_df['y'] = p_df['y'].map({'no':0, 'yes':1})# Encode the targeted variable


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

# MARKDOWN ********************

# ##### Steps to analyze the categorical variables
#     1. Get the unique categories 
#     2. Group and Sum them up by category
#     3. Test for independence using chi square test - between the independent variables and the target variable
#     4. Stratified sampling from each categories

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

y_yes = cat_col_df.loc[p_df['y']=='yes']
y_yes

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

y_no = cat_col_df.loc[p_df['y']=='no']
y_no

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

col_names = cat_col_df.columns.tolist()[:-1]
target_col = cat_col_df.columns.tolist()[-1]
target_col

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def count_cat(df,colname, target_column):
    result = pd.crosstab(df[colname], df[target_column])
    return result

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def dependent_or_not(df, col_name, target_col, p_value):
    if p_value <= 0.05:
        print(f'P value: {round(p_value,4)}')
        print(" ")
        print(f'{col_name[0]} and {df.target_col} are significantly related')
    else:
        print(f"{df.col_name} and {df.target_col} are independent")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

### counts of the actual observed outcome of each categorical columns


job_count = count_cat(cat_col_df,col_names[0], 'y')
marital_count = count_cat(cat_col_df,col_names[1], 'y')
education_count = count_cat(cat_col_df,col_names[2], 'y')
default_count = count_cat(cat_col_df,col_names[3], 'y')
housing_count = count_cat(cat_col_df,col_names[4], 'y')
loan_count = count_cat(cat_col_df,col_names[5], 'y')
contact_count = count_cat(cat_col_df,col_names[6], 'y')
month_count_count = count_cat(cat_col_df,col_names[7], 'y')
day_of_week_count = count_cat(cat_col_df,col_names[8], 'y')
poutcome_count = count_cat(cat_col_df,col_names[9], 'y')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

month_count_count # month variable can affect the model so use this col to statify

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

poutcome_count

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def get_chi_pvalues(df, indep_cols, target):
    p_value_dict = dict()
    for col in indep_cols:
        # contingency table
        ct = pd.crosstab(df[col], df[target])

        # chi-square
        chi2, p_value, dof, expected = chi2_contingency(ct)

        #save in dict
        p_value_dict[col]= p_value
    return p_value_dict

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#convert the dict to df
p_dict = get_chi_pvalues(cat_col_df, col_names, target_col) 
#p_df = pd.DataFrame.from_dict(p_dict, orient='index', columns=["p_value"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# create a df
p_value_df = pd.DataFrame.from_dict(p_dict, orient='index', columns=['p_value'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

p_value_df['-log10_p'] = -np.log10(p_value_df['p_value'] + 1e-300)
p_value_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# plot a heatmap

plt.figure(figsize=(5,8))
sns.heatmap(
        p_value_df[['-log10_p']],
        annot=p_value_df[['p_value']],
        fmt='.4f',
        cmap='Blues',
        cbar_kws={'label': '-log10(p_value)'}    
        )
plt.title('Variables Significace Heatmap')
plt.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Model training preparation


# CELL ********************

# combine the numerical and categroical dfs

df_num = spark.read.format('delta').load('abfss://project1_bank_market@onelake.dfs.fabric.microsoft.com/project1_lakehouse.Lakehouse/Tables/dbo/num_table_train')
df_num_train = df_num.toPandas()
df_num_train

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# add the categorical variables excluding the target

df_model = pd.concat([df_num_train, cat_col_df.drop(['y'], axis=1)], axis=1)
df_model

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Split the data for validation and training

# Combined key
df_model['stratify_key']=df_model['month'].astype(str) + "_" + df_model['y'].astype(str)

# Split the data with the composite key
train_df, val_df = train_test_split(df_model, test_size=0.2, stratify=df_model['stratify_key'], random_state=42)

train_df = train_df.drop(columns=['stratify_key'])
val_df = val_df.drop(columns=['stratify_key'])



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# upsampling the training data

# group by month and target column and find the highest class and up sample the lowest class to match

target_count = train_df.groupby(['month','y']).size()
max_per_month = target_count.groupby('month').max()

# Resample each target group 
df_train = train_df.groupby(['month', 'y'], group_keys=False).apply(
     lambda group: group.sample(max_per_month[group.name[0]], replace=True)
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_train

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# X and y for the train and val 

X_train = df_train.drop('y', axis=1)
y_train = df_train['y']

X_val = val_df.drop('y', axis=1)
y_val = val_df['y']

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

# # Min max normalization for the numeric columns - preserve the original distribution
# column names
nums_col = X_train.select_dtypes(include=[int, float]).columns.to_list()

scaler = MinMaxScaler()

# X_train
X_train[nums_col] = scaler.fit_transform(X_train[nums_col])

# X_val
X_val[nums_col] = scaler.fit_transform(X_val[nums_col])

X_train.head(5)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Hotencode the categorical variables 
# Isolate columns
hot_col_cat = X_train.select_dtypes(include=object).columns.to_list()
col_num = X_train.select_dtypes(include=[int,float]).columns.to_list()



cat_encoder = OneHotEncoder(sparse_output=False, drop='first').set_output(transform="pandas")

# encode X_train 
X_train_enc = cat_encoder.fit_transform(X_train[hot_col_cat])

#encode X_val
X_val_enc = cat_encoder.fit_transform(X_val[hot_col_cat])


# combaine the numerical and categorical cols

X_train_final = pd.concat([X_train[col_num],X_train_enc], axis=1)

X_val_final = pd.concat([X_val[col_num], X_val_enc], axis=1)




# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Save the files 
# X_train_final and y_train

X_train_final.write.format('delta').option('overwriteSchema', 'false').saveAsTable('X_train_final')
y_train.write.format('delta').option('overwriteSchema', 'false').saveAsTable('y_train')

# X_val_final and y_val
X_val_final.write.format('delta').option('overwriteSchema', 'false').saveAsTable('X_val_final')
y_val.write.format('delta').option('overwriteSchema', 'false').saveAsTable('y_val')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

y_val

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#get the numeric columns
X_train_num = X_train[col_num].to_numpy()
X_val_num = X_val[col_num].to_numpy()

#Combine numeric array and encoded array 
X_train_final = np.hstack((X_train_num, X_train_enc))

X_val_final = np.hstack((X_val_num, X_val_enc))

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

X_val_final

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# CELL ********************

job_col = dependent_or_not(p_value )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

p_value.round(2),expected.round(0)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

job_count.insert(2,'expected_no', expected[:,:-1].round(0))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

job_count.insert(2,'expected_yes', expected[:,1:].round(0))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

job_count = job_count[['no','expected_no', 'yes', 'expected_yes']]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

job_count

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
