# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Data.csv')


# change the Categorical variables,
# =====================================================================================

# step-1: change into nominal categories, let say purchase column with 2 cats:
# ---------------------------------------------------------------------------
# repeat for all text of type nomical
print(df.Purchased)      # look at the dtype
df.Purchased = pd.Categorical(df.Purchased,categories=["No","Yes"])
print(df.Purchased)      # look at the dtype
df.Purchased = df.Purchased.cat.codes


#Step-2: get dummies remove dummy variable trap (i.e. 1 less variable )
# -------------------------------------------------------------------
tmp = pd.get_dummies(df)  #--> get dummies for rest of them
df = tmp.iloc[:,[4,5,6,0,1,2,3]]  # --> full data

X = df.iloc[:,[1,2,3,4,5]].values  # first dummy is removed
y = df.Profit.values




X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
