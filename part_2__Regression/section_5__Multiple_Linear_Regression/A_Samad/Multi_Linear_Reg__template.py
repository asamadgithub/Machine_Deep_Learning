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


# Feature Scaling, if library is not providing that bey-default:
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



# Simple Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)


y_pred = lin_reg.predict(X_test)


# Plotting
#===============================
#plotly.offline doesn't push your charts to the clouds
import plotly.offline as pyo

#allows us to create the Data and Figure objects
from plotly.graph_objs import *

#plotly.plotly pushes your charts to the cloud
# import plotly.plotly as py (this is depreciated)
# import chart_studio.plotly as py

#pandas is a data analysis library
#from pandas import DataFrame

#lets us see the charts in an iPython Notebook
pyo.offline.init_notebook_mode() # run at the start of every ipython

trace_1 = {
	"type"   :   "scatter",
		"mode"   :   "markers",
			"name"   :   "real_values",
    "x"      :    X_test.ravel(),
    "y"      :    y_test,
    "hoverinfo"  :      "x+y",
}

trace_2 = {
	"type"   :   "scatter",
		"mode"   :   "markers",
			"name"   :   "Predicted_values",
    "x"      :    X_test.ravel(),
    "y"      :    y_pred,
    "hoverinfo"  :      "x+y",
}
data = [trace_1,trace_2]

graph_layout = {
	"title":"Linear Regression Model",
		"xaxis":{"title":"X_test values"},
			"yaxis":{"title":"Salaries"},
}
fig = Figure(data=data, layout=graph_layout)
pyo.iplot(fig)



import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis = 1)  # add column of ones


# 1):  Select a significance level   (SL = 0.05) <br>
# 2):  Fit the model with all possible predictors  <br>
# 3):  Condier the predictor with highest p-values, if P> S.L., ==> step-4, else FINISH  <br>
# 4):  Remove the predictor   <br>
# 5):  Fit the model without this variable ==> step-3   <br>


 print(np.array_str(X[1:4], precision = 3, suppress_small = True))
X_opt = X[:,[0,1,2,3,4,5]]    # full span and then gradually reduce
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


 print(np.array_str(X[1:4], precision = 3, suppress_small = True))
# Redefine the X_opt
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
...
...
...







