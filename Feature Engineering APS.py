import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df_og = pd.read_csv('train.csv')
df = df_og.copy()
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
df['satisfaction'] = df.satisfaction.apply(lambda x: int(1) if x == 'satisfied' else int(0))

# fix the NAN and Not Applicable data
NA_columns = [column for column in df.columns if df[column].isnull().sum() > 0]
## fill the NA in Arrival Delay in Minutes with the median value
for col in NA_columns:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
## check the NA after the fillna
df.isnull().sum()
## transform the Not Applicable (0) with the modus in columns services (num_columns_discrete)
num_columns = [column for column in df.columns if df[column].dtypes != 'O']
num_columns_discrete = [column for column in num_columns if len(df[column].unique()) <= 10 and column not in ['satisfaction']]
for col in num_columns_discrete:
    modus_value = str(df[col].mode()[0])
    df[col] = df[col].astype(str).apply(lambda x: x.replace('0', modus_value))
    df[col] = df[col].astype(int)
    
# encode string categorical column into numeric
cat_columns = [column for column in df.columns if column not in num_columns]
def encode_category(data, column, target):
    ordinal_data= data.groupby([column])[target].sum().sort_values(by=column, ascending=False).index
    ordinal_num = {k: i for i, k in enumerate(ordinal_data, start=0)}
    data[column] = data[column].map(ordinal_num)
    print(ordinal_data)
    
for col in cat_columns:
    encode_category(df, col, ['satisfaction'])
## the encode is ordered by the sum of satisfaction within one categoy in each columns

# Scale the dataset
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
columns_training = [column for column in df.columns if column not in ['satisfaction']]
df[columns_training] = sc_X.fit_transform(df[columns_training])

# save the dataset from the Feature Engineering Process
df.to_csv('Train Feature Engineering.csv', index=False)
