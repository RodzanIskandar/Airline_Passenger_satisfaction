import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Train Feature Engineering.csv')

y = df['satisfaction']
X = df.drop(['satisfaction'], axis=1)

# Filter Methods
# Continuous Columns
cont_columns = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
sns.heatmap(df[cont_columns].corr(), annot=True)
plt.tight_layout()
## 'Departure Delay in Minutes' and 'Arrival Delay in Minutes' are too corelated for the correlation between independent features, so one of the features need to be drop
from sklearn.feature_selection import f_classif, SelectKBest
selector_cont = SelectKBest(score_func = f_classif, k=3)
selector_cont.fit(X[cont_columns], y)
pd.DataFrame({'Features':X[cont_columns].columns, 'F-Score':selector_cont.scores_, 'p-value':selector_cont.pvalues_})
## cause the Departure Delay in minutes is the smallest F-score, so I drop it.
cont_select = X[cont_columns].columns[selector_cont.get_support()].tolist()

# Categorical columns
## for categorical columns I use chi squared function.
cat_columns = [column for column in X.columns if column not in cont_columns]
from sklearn.feature_selection import chi2
selector_cat = SelectKBest(score_func= chi2, k=15)
selector_cat.fit(X[cat_columns], y)
pd.DataFrame({'Features':X[cat_columns].columns, 'score':selector_cat.scores_, 'p-value':selector_cat.pvalues_})
## from the score, Gender, Departure/Arrival time convenient, and gate location are the less important feature and have big score difference with another features
cat_select = X[cat_columns].columns[selector_cat.get_support()].tolist()

# All selected features
selected_features = cont_select + cat_select

# save the all selected features into csv file
pd.Series(selected_features).to_csv('selected_features.csv', index=False)
