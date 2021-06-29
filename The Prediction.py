import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df_og = pd.read_csv('test.csv')
df = df_og.copy()

# The Prediction is treated by same proceses and same variables

df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
df['satisfaction'] = df.satisfaction.apply(lambda x: int(1) if x == 'satisfied' else int(0))

NA_columns = ['Arrival Delay in Minutes']
for col in NA_columns:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    
num_columns_discrete = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service',
                        'Cleanliness']
for col in num_columns_discrete:
    modus_value = str(df[col].mode()[0])
    df[col] = df[col].astype(str).apply(lambda x: x.replace('0', modus_value))
    df[col] = df[col].astype(int)
    
df['Flight Distance'] = np.log(df['Flight Distance'])


ordinal_cat_columns = {'Gender': {'Male':0, 'Female':1},
                       'Customer Type': {'disloyal Customer': 0, 'Loyal Customer':1},
                       'Type of Travel': {'Personal Travel':0, 'Business travel':1},
                       'Class': {'Eco Plus':0, 'Eco':1, 'Business':2}}
for col in ordinal_cat_columns.keys():
    label = ordinal_cat_columns[col]
    df[col] = df[col].map(label)

import pickle
scaler = pickle.load(open('scaler_APS.pickle', 'rb'))
columns_training = [column for column in df.columns if column not in ['satisfaction']]
df[columns_training] = scaler.transform(df[columns_training])
    
features = pd.read_csv('selected_features.csv')
features = features['0'].tolist()

df = df[features]

ml_model = pickle.load(open('Airline_Passenger_satisfaction.pickle', 'rb'))
prediction = ml_model.predict(df)


'''
y_test = df_og['satisfaction']
y_test = y_test.apply(lambda x: int(1) if x == 'satisfied' else int(0))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)
'''

