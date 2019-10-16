import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.compose import TransformedTargetRegressor
import pickle
import sys

#Load
dataset = pd.read_csv('/Users/grebelsm/Documents/TCD Final Year/CSU44061 Machine Learning/Individual Competition/data/tcd ml 2019-20 income prediction training (with labels).csv')

#Choose features + label
dataset = dataset[['Year of Record', 'Gender', 'Age', 'Country', 'Size of City', 'Profession', 'University Degree', 'Body Height [cm]', 'income']]

#Fills Age NAs with previous age
print(dataset.isna().sum())
dataset["Age"].fillna( method ='ffill', inplace = True)
dataset["Country"].fillna( method ='ffill', inplace = True)
dataset["Profession"].fillna( method ='ffill', inplace = True)
dataset["Year of Record"].fillna( method ='ffill', inplace = True)
dataset["Gender"].fillna( method ='ffill', inplace = True)
dataset.dropna(inplace = True)

Y = dataset['income']
#Encode (Hot & Label)
dataset = pd.get_dummies(dataset, prefix_sep='_', drop_first=True)
le = LabelEncoder()
dataset = dataset.apply(le.fit_transform)

#Follow normal distribution
tf = QuantileTransformer(output_distribution='normal')

#Drop label
X = dataset.drop('income', 1)


print(X.shape)


#Train
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = LinearRegression()
#Transform
regressor = TransformedTargetRegressor(regressor=regressor, transformer=tf)  
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(x_test)


#Score me
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#back it up
filename = 'submission.sav'
pickle.dump(regressor, open(filename, 'wb'))










#Load without labels
dataset = pd.read_csv('/Users/grebelsm/Documents/TCD Final Year/CSU44061 Machine Learning/Individual Competition/data/tcd ml 2019-20 income prediction test (without labels).csv')

#Pick features minus income
dataset = dataset[['Year of Record', 'Gender', 'Age', 'Country', 'Size of City', 'Profession', 'University Degree', 'Body Height [cm]']]

#Fill NAs with previous entry
dataset["Age"].fillna( method ='ffill', inplace = True)
dataset["Country"].fillna( method ='ffill', inplace = True)
dataset["Profession"].fillna( method ='ffill', inplace = True)
dataset["Year of Record"].fillna( method ='ffill', inplace = True)
dataset["Gender"].fillna( method ='ffill', inplace = True)
#Encode
dataset = pd.get_dummies(dataset, prefix_sep='_', drop_first=True)

dataset = dataset.apply(le.fit_transform)

#Match columns with trained model
X, dataset = X.align(dataset, join='left', axis=1)



print(dataset.isna().sum())
print(dataset.head())
dataset.fillna(value=0, inplace=True)
print(dataset.isna().sum())
print(dataset.shape)

#Load + predict
loaded_model = pickle.load(open('submission.sav', 'rb'))
result = loaded_model.predict(dataset)
np.savetxt('submission.csv', result, delimiter=',')

